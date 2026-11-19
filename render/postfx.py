"""Post-processing chain planning for an HDR renderer.

An effect is passed inline as a dict::

    {
        "name": "bloom",
        "inputs": ["scene", "bloom_tmp"],
        "outputs": ["bloom_tmp"],
        "format": "rgba16f",
        "scale": 0.5,                       # optional, default 1.0
        "sample_rect": [0, 0, 1920, 1080],  # optional pixel read rect
    }

The four required keys are exactly ``name``, ``inputs``, ``outputs`` and
``format``.  ``inputs`` and ``outputs`` contain logical render-target ids;
``format`` is ``"rgba16f"`` for HDR data or ``"rgba8"`` for LDR data.  An
effect is one GPU pass, and list order is pass order.  Optional ``scale``
selects a resolution multiplier (``scale > 0``) and optional ``sample_rect``
describes the source pixels read by that pass.

A *caps* mapping describes the device budget::

    {
        "max_targets": 6,
        "supported_formats": ("rgba16f", "rgba8"),
        "base_size": (1920, 1080),
    }

All keys are optional where documented by the individual helper.  Every
helper here is deterministic, uses only the Python standard library, and never
reads the clock, the network or random state.
"""

import heapq
import math
from collections.abc import Mapping


class PostFxError(Exception):
    """Raised for an invalid chain, cycle, pool exhaustion or budget denial."""


_FORMATS = ("rgba16f", "rgba8")

_REQUIRED_KEYS = ("name", "inputs", "outputs", "format")


def _check_effect(effect, index):
    """Raise PostFxError unless ``effect`` is a well-formed effect dict."""
    where = "effect #%d" % index
    if not isinstance(effect, dict):
        raise PostFxError("%s is not a dict" % where)
    for key in _REQUIRED_KEYS:
        if key not in effect:
            raise PostFxError("%s is missing required key %r" % (where, key))
    if not isinstance(effect["name"], str) or not effect["name"]:
        raise PostFxError("%s has an invalid name" % where)
    for key in ("inputs", "outputs"):
        value = effect[key]
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            raise PostFxError("%s has non-string-list %r" % (where, key))
    if not isinstance(effect["format"], str):
        raise PostFxError("%s has a non-string format" % where)


def _check_chain(chain):
    if not isinstance(chain, list):
        raise PostFxError("chain must be a list of effect dicts")
    for index, effect in enumerate(chain):
        _check_effect(effect, index)
    names = [effect["name"] for effect in chain]
    if len(set(names)) != len(names):
        raise PostFxError("duplicate effect name in chain")


def _dependency_edges(effects):
    """Return (names, deps) where deps[name] is the set of producer names
    that ``name`` depends on.  Self-edges (a pass reading its own output)
    are ignored: that pattern is an in-pass hazard, not an ordering edge."""
    names = [effect["name"] for effect in effects]
    producer = {}
    for effect in effects:
        for out in effect["outputs"]:
            if out in producer and producer[out] != effect["name"]:
                raise PostFxError(
                    "render target %r has more than one producer" % out
                )
            producer[out] = effect["name"]
    deps = {name: set() for name in names}
    for effect in effects:
        for source in effect["inputs"]:
            owner = producer.get(source)
            if owner is not None and owner != effect["name"]:
                deps[effect["name"]].add(owner)
    return names, deps


def _topological_order(effects):
    """Iterative Kahn's algorithm, stable by registration order.

    Returns the ordered list of names, or ``None`` when the graph contains
    a cycle.  Never recurses, so cyclic input cannot overflow the stack.
    """
    names, deps = _dependency_edges(effects)
    consumers = {name: [] for name in names}
    indegree = {}
    for name in names:
        indegree[name] = len(deps[name])
        for parent in deps[name]:
            consumers[parent].append(name)
    position = {name: i for i, name in enumerate(names)}
    ready = [position[name] for name in names if indegree[name] == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        index = heapq.heappop(ready)
        name = names[index]
        order.append(name)
        for consumer in consumers[name]:
            indegree[consumer] -= 1
            if indegree[consumer] == 0:
                heapq.heappush(ready, position[consumer])
    if len(order) != len(names):
        return None
    return order


def _last_read_indices(chain):
    """Map each logical RT id to the last chain index that reads it."""
    last_read = {}
    for index, effect in enumerate(chain):
        for source in effect["inputs"]:
            last_read[source] = index
    return last_read


def _peak_live_targets(chain):
    """Peak number of simultaneously live temporary targets.

    A target is live from the pass that produces it until the last pass
    that reads it (inclusive); an unread target dies with its producer.
    """
    last_read = _last_read_indices(chain)
    peak = 0
    live = []
    for index, effect in enumerate(chain):
        live = [entry for entry in live if entry[1] >= index]
        for out in effect["outputs"]:
            live.append((out, max(last_read.get(out, index), index)))
        peak = max(peak, len(live))
    return peak


def _is_positive_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_positive_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value > 0
    )


def build_chain(effects: list[dict], caps: dict) -> list[str]:
    """Return effect names in a correct, deterministic execution order.

    Dependencies are inferred from logical render targets.  Effect ``B``
    depends on effect ``A`` when one of ``B["inputs"]`` appears in
    ``A["outputs"]``; external names (for example ``"scene"``) have no producer
    and impose no edge.  Every producer therefore precedes its consumers.

    The topological order is stable: among effects whose prerequisites have all
    been emitted, the one that appeared first in ``effects`` is emitted first.
    A chain registered in reverse dependency order therefore executes in
    dependency order, not registration order.

    ``caps`` uses the mapping documented at module level.  Ordering does not
    depend on budget or format policy; those checks belong to
    :func:`validate_chain`.

    Args:
        effects: unordered effect dicts; not mutated.
        caps: device capability/budget mapping; not mutated.

    Returns:
        A new list of effect names.  Empty input returns ``[]``.

    Raises:
        PostFxError: the input is malformed, an effect name is duplicated, an
            output RT has more than one producer, or the dependency graph
            contains a cycle.  Cycle detection is iterative, so a cyclic input
            raises instead of causing ``RecursionError``.
    """
    if not isinstance(effects, list):
        raise PostFxError("effects must be a list")
    if not effects:
        return []
    _check_chain(effects)
    order = _topological_order(effects)
    if order is None:
        raise PostFxError("dependency cycle detected among effects")
    return order


def allocate_targets(chain: list[dict], pool: list[str], format_pref: str) -> dict:
    """Assign reusable physical render targets to an ordered chain.

    ``chain`` is a list of effect dicts in execution order, normally the
    effects corresponding to the names returned by :func:`build_chain`.
    ``pool`` is a list of distinct physical RT ids supplied by the caller.
    Every target in the pool is assumed to have the size and ``format_pref``
    required by the effects.  An effect's logical output ``outputs[0]`` is
    mapped one-to-one to a physical target in ``"assign"``.

    Targets are reused temporally.  A producer's target stays live until the
    last later effect that reads that logical output has executed; only then
    may the physical target be assigned again.  A target currently read by an
    effect is not chosen for that same effect's output (in-pass aliasing is a
    separate hazard handled by :func:`insert_barriers`).  Allocation is first
    fit in ``pool`` order, making the result deterministic.  For a serial
    chain two targets are enough (ping-pong), regardless of chain length.

    Args:
        chain: ordered effect dicts; not mutated.
        pool: distinct physical RT ids available for temporary work.
        format_pref: required pool format, ``"rgba16f"`` or ``"rgba8"``.

    Returns:
        ``{"assign": {effect_name: physical_rt_id}, "count": int}`` where
        ``count`` is the number of distinct physical targets used, not the
        number of effects.  An empty chain returns
        ``{"assign": {}, "count": 0}``.

    Raises:
        PostFxError: the chain or pool is malformed, ``format_pref`` is not
            supported, the pool contains duplicates, or no free target exists.
    """
    if not isinstance(pool, list) or not all(
        isinstance(item, str) for item in pool
    ):
        raise PostFxError("pool must be a list of physical RT ids")
    if len(set(pool)) != len(pool):
        raise PostFxError("pool contains duplicate target ids")
    if format_pref not in _FORMATS:
        raise PostFxError("unsupported pool format %r" % (format_pref,))
    if not isinstance(chain, list):
        raise PostFxError("chain must be a list of effect dicts")
    if not chain:
        return {"assign": {}, "count": 0}
    _check_chain(chain)

    last_read = _last_read_indices(chain)
    busy = {}  # physical id -> chain index after which it becomes free
    used = set()
    assign = {}
    for index, effect in enumerate(chain):
        if not effect["outputs"]:
            continue
        chosen = None
        for physical in pool:
            if physical not in busy or busy[physical] < index:
                chosen = physical
                break
        if chosen is None:
            raise PostFxError(
                "render target pool exhausted at effect %r" % effect["name"]
            )
        last_use = index
        for out in effect["outputs"]:
            last_use = max(last_use, last_read.get(out, index))
        busy[chosen] = last_use
        used.add(chosen)
        assign[effect["name"]] = chosen
    return {"assign": assign, "count": len(used)}


def resolve_format(effects: list[dict], output_format: str) -> str:
    """Resolve the working format and enforce a single conversion at the end.

    Intermediate passes must remain ``"rgba16f"`` so HDR values survive until
    tone mapping and presentation.  An ``"rgba8"`` effect anywhere except the
    final position is an early conversion and is rejected.

    ``output_format`` is the format requested by the caller.  For
    ``"rgba8"``, only the last pass writes 8-bit data and this function returns
    ``"rgba8"``; every earlier effect must still declare ``"rgba16f"``.  For
    ``"rgba16f"``, the whole chain remains HDR and the function returns
    ``"rgba16f"``; a final ``"rgba8"`` effect is then a mismatch.  An empty
    chain returns ``output_format`` unchanged.

    Args:
        effects: ordered effect dicts, final pass last; not mutated.
        output_format: requested final format, ``"rgba16f"`` or ``"rgba8"``.

    Returns:
        ``"rgba16f"`` or ``"rgba8"``: the format the last pass must write.

    Raises:
        PostFxError: an unsupported format appears, 8-bit conversion occurs
            before the final pass, or an 8-bit final pass contradicts an HDR
            request.
    """
    if output_format not in _FORMATS:
        raise PostFxError("unsupported output format %r" % (output_format,))
    if not isinstance(effects, list):
        raise PostFxError("effects must be a list")
    _check_chain(effects)
    if not effects:
        return output_format
    for effect in effects:
        if effect["format"] not in _FORMATS:
            raise PostFxError(
                "effect %r uses unsupported format %r"
                % (effect["name"], effect["format"])
            )
    for effect in effects[:-1]:
        if effect["format"] != "rgba16f":
            raise PostFxError(
                "effect %r converts to 8-bit before the final pass; "
                "HDR data would be crushed before tone mapping"
                % effect["name"]
            )
    final_format = effects[-1]["format"]
    if final_format != output_format:
        raise PostFxError(
            "final pass %r writes %r but the requested output format is %r"
            % (effects[-1]["name"], final_format, output_format)
        )
    return output_format


def insert_barriers(chain: list[dict]) -> list[tuple]:
    """Return the barriers required by an ordered chain.

    One effect is one pass, so a pass that both reads and writes the same RT
    needs a read/write barrier *inside* that pass; that is exactly the hazard a
    cross-pass-only implementation misses.  Cross-pass read-after-write and
    write-after-read hazards are reported too.

    A barrier tuple is ``(effect_name, rt_id, hazard)`` and means "insert this
    barrier immediately before executing effect_name".  ``hazard`` is one of
    ``"rw"`` (same-pass read/write of the same RT), ``"raw"`` (an earlier pass
    wrote the RT and this pass reads it), or ``"war"`` (an earlier pass read the
    RT and this pass writes it).

    Args:
        chain: ordered effect dicts; not mutated.

    Returns:
        A deduplicated list of barrier tuples in execution order.  Empty input
        returns ``[]``.

    Raises:
        PostFxError: the chain is malformed.
    """
    if not isinstance(chain, list):
        raise PostFxError("chain must be a list of effect dicts")
    _check_chain(chain)
    barriers = []
    seen = set()
    written = set()
    read = set()
    for effect in chain:
        inputs = list(dict.fromkeys(effect["inputs"]))
        outputs = list(dict.fromkeys(effect["outputs"]))
        for rt in inputs:
            if rt in outputs:
                hazard = "rw"
            elif rt in written:
                hazard = "raw"
            else:
                continue
            key = (effect["name"], rt, hazard)
            if key not in seen:
                seen.add(key)
                barriers.append(key)
        for rt in outputs:
            if rt in inputs:
                continue  # already covered by the in-pass "rw" barrier
            if rt in read:
                key = (effect["name"], rt, "war")
                if key not in seen:
                    seen.add(key)
                    barriers.append(key)
        written.update(outputs)
        read.update(inputs)
    return barriers


def scale_targets(
    chain: list[dict], base_w: int, base_h: int, scale: float
) -> dict:
    """Return per-effect output dimensions for a resolution scale.

    The effective scale is ``effect["scale"]`` when present, otherwise the
    global ``scale``.  Dimensions are ``max(1, ceil(base * scale))`` on each
    axis.  Ceiling is required: a 101-pixel base at 0.5 is 51, not 50, so an
    odd row or column is never dropped.  ``scale=1.0`` reproduces the base size
    exactly, including odd dimensions.

    Args:
        chain: ordered effect dicts; optional per-effect ``scale``.
        base_w: native frame width in pixels, positive.
        base_h: native frame height in pixels, positive.
        scale: global scale, positive.

    Returns:
        ``{effect_name: (width, height)}`` with integer pixel sizes.

    Raises:
        PostFxError: a dimension or scale is not positive, or the chain is
            malformed.
    """
    if not isinstance(chain, list):
        raise PostFxError("chain must be a list of effect dicts")
    if not _is_positive_int(base_w) or not _is_positive_int(base_h):
        raise PostFxError("base dimensions must be positive integers")
    if not _is_positive_number(scale):
        raise PostFxError("scale must be a positive number")
    _check_chain(chain)
    sizes = {}
    for effect in chain:
        effective = effect.get("scale", scale)
        if not _is_positive_number(effective):
            raise PostFxError(
                "effect %r has a non-positive scale" % effect["name"]
            )
        width = max(1, math.ceil(round(base_w * effective, 9)))
        height = max(1, math.ceil(round(base_h * effective, 9)))
        sizes[effect["name"]] = (width, height)
    return sizes


def merge_passes(a: dict, b: dict) -> bool:
    """Return whether two adjacent passes can be merged without changing data.

    Passes merge only when they read exactly the same set of RTs, write exactly
    the same set of RTs, use the same format, and have no barrier conflict.
    Format equality is mandatory: an ``"rgba16f"`` pass cannot absorb an
    ``"rgba8"`` pass (or vice versa).  A barrier conflict exists when either
    pass both reads and writes an RT, or a target read by one pass is written
    by the other.

    Args:
        a: first pass effect dict.
        b: second adjacent pass effect dict.

    Returns:
        ``True`` only when the passes are semantically identical and safe to
        fuse; otherwise ``False``.

    Raises:
        PostFxError: either argument is not a well-formed effect dict.
    """
    _check_effect(a, 0)
    _check_effect(b, 1)
    if a["format"] != b["format"]:
        return False
    a_in, a_out = set(a["inputs"]), set(a["outputs"])
    b_in, b_out = set(b["inputs"]), set(b["outputs"])
    if a_in != b_in or a_out != b_out:
        return False
    if a_in & a_out or b_in & b_out:
        return False  # a pass that reads and writes the same RT
    if a_in & b_out or b_in & a_out:
        return False  # one pass writes a target the other reads
    return True


def validate_chain(chain: list[dict], caps: dict) -> list[str]:
    """Return a list of human-readable problems with a chain.

    An empty list means the chain is valid.  Each problem starts with a stable
    prefix followed by ``": "``:

    * ``"cycle: "`` -- the dependency graph contains a cycle; detection is
      iterative.
    * ``"budget: "`` -- peak simultaneously live temporary targets exceed
      ``caps["max_targets"]``.
    * ``"format: "`` -- an unsupported format or an early ``"rgba8"``
      conversion, according to ``caps["supported_formats"]``.
    * ``"sample: "`` -- an effect's ``sample_rect`` is malformed or falls
      outside ``caps["base_size"]``.  Out-of-bounds reads stretch edge pixels.

    The function collects all independent problems rather than stopping at the
    first.  ``caps`` keys are optional: without ``max_targets`` the budget is
    not checked, without ``base_size`` sample rects cannot be checked, and
    ``supported_formats`` defaults to both documented formats.

    Args:
        chain: effect dicts, ordered or unordered; not mutated.
        caps: device capability/budget mapping.

    Returns:
        A deterministically ordered list of issue strings.

    Raises:
        PostFxError: only for a malformed argument container; chain defects are
            returned as issues, never raised.
    """
    if not isinstance(chain, list) or not isinstance(caps, Mapping):
        raise PostFxError("chain must be a list and caps a mapping")
    _check_chain(chain)
    issues = []

    if _topological_order(chain) is None:
        issues.append("cycle: dependency graph contains a cycle")

    max_targets = caps.get("max_targets")
    if max_targets is not None:
        peak = _peak_live_targets(chain)
        if peak > max_targets:
            issues.append(
                "budget: chain needs %d simultaneous targets but "
                "max_targets is %d" % (peak, max_targets)
            )

    supported = caps.get("supported_formats", _FORMATS)
    last_index = len(chain) - 1
    for index, effect in enumerate(chain):
        fmt = effect["format"]
        if fmt not in supported:
            issues.append(
                "format: effect %r uses unsupported format %r"
                % (effect["name"], fmt)
            )
        elif fmt == "rgba8" and index != last_index:
            issues.append(
                "format: effect %r converts to rgba8 before the final pass"
                % effect["name"]
            )

    base_size = caps.get("base_size")
    if base_size is not None:
        base_w, base_h = base_size
        for effect in chain:
            if "sample_rect" not in effect:
                continue
            rect = effect["sample_rect"]
            malformed = (
                not isinstance(rect, (list, tuple))
                or len(rect) != 4
                or not all(isinstance(v, (int, float)) for v in rect)
            )
            if malformed:
                issues.append(
                    "sample: effect %r has a malformed sample_rect"
                    % effect["name"]
                )
                continue
            x, y, w, h = rect
            if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > base_w or y + h > base_h:
                issues.append(
                    "sample: effect %r reads rect %r outside the %dx%d frame"
                    % (effect["name"], list(rect), base_w, base_h)
                )
    return issues


def run_frames(frames: list[str], chain: list[dict], budget: dict) -> dict:
    """Execute a chain for several frames with per-frame RT state.

    ``frames`` is an ordered list of opaque frame ids.  Execution is symbolic:
    every render-target value is tagged with the id of the frame that produced
    it, so a stale value from an earlier frame is observable instead of being
    hidden.  Each frame starts from a clean logical RT table.  Physical target
    ids may be reused, but no assignment or tagged value may survive from one
    frame to the next.  The target count is a per-frame peak, so it does not
    grow with the number of frames.

    ``budget`` is ``{"max_targets": int}``.  If the chain needs more
    simultaneous targets than the budget allows, no frame is executed and the
    whole request fails explicitly.  The implementation must not silently
    bypass post-processing (no ``"pass_through"`` or ``"degraded"`` mode).

    Args:
        frames: ordered frame ids; each id is unique.
        chain: ordered effect dicts; not mutated.
        budget: mapping with a non-negative integer ``max_targets``.

    Returns:
        On success::

            {
              "ok": True,
              "rt_count": int,          # peak distinct targets in one frame
              "frames": [
                {
                  "frame": frame_id,
                  "generation": frame_index,
                  "targets": {effect_name: physical_rt_id},
                  "outputs": {logical_rt_id: frame_id},
                },
                ...
              ],
            }

        Every ``outputs`` value equals that record's frame id.  On budget
        denial::

            {"ok": False, "skipped": [every frame id], "reason": "budget"}

        with no ``"frames"`` list and no degraded mode.

    Raises:
        PostFxError: the chain, frames or budget is malformed, a frame id is
            duplicated, or the chain is cyclic.
    """
    if not isinstance(frames, list) or not isinstance(chain, list):
        raise PostFxError("frames and chain must be lists")
    if not isinstance(budget, Mapping):
        raise PostFxError("budget must be a mapping")
    max_targets = budget.get("max_targets")
    if (
        not isinstance(max_targets, int)
        or isinstance(max_targets, bool)
        or max_targets < 0
    ):
        raise PostFxError("budget requires a non-negative integer max_targets")
    if len(set(frames)) != len(frames):
        raise PostFxError("duplicate frame id")
    _check_chain(chain)
    if _topological_order(chain) is None:
        raise PostFxError("dependency cycle detected among effects")

    needed = _peak_live_targets(chain)
    if needed > max_targets:
        return {"ok": False, "skipped": list(frames), "reason": "budget"}

    pool = ["rt-%d" % i for i in range(max(needed, 1))]
    allocation = allocate_targets(chain, pool, "rgba16f")
    rt_count = allocation["count"]

    records = []
    for generation, frame_id in enumerate(frames):
        # Fresh tables per frame: no assignment or tagged value survives.
        targets = dict(allocation["assign"])
        outputs = {}
        for effect in chain:
            for out in effect["outputs"]:
                outputs[out] = frame_id
        records.append(
            {
                "frame": frame_id,
                "generation": generation,
                "targets": targets,
                "outputs": outputs,
            }
        )
    return {"ok": True, "rt_count": rt_count, "frames": records}
