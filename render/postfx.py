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


class PostFxError(Exception):
    """Raised for an invalid chain, cycle, pool exhaustion or budget denial."""


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
