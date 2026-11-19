"""Behavioural tests for the HDR post-processing chain planner.

Every test is red on the shipped stubs: each one calls the public API and the
only failure mode is ``NotImplementedError``.  Do not edit or weaken the tests.
"""

import pytest

from render.postfx import (
    PostFxError,
    allocate_targets,
    build_chain,
    insert_barriers,
    merge_passes,
    resolve_format,
    run_frames,
    scale_targets,
    validate_chain,
)


def test_defect_01_build_chain_orders_by_dependency_not_registration():
    effects = [
        {
            "name": "fxaa",
            "inputs": ["tone_out"],
            "outputs": ["ldr"],
            "format": "rgba8",
        },
        {
            "name": "bloom",
            "inputs": ["scene"],
            "outputs": ["bloom_out"],
            "format": "rgba16f",
        },
        {
            "name": "tone",
            "inputs": ["bloom_out"],
            "outputs": ["tone_out"],
            "format": "rgba16f",
        },
    ]
    caps = {
        "max_targets": 4,
        "supported_formats": ("rgba16f", "rgba8"),
        "base_size": (1920, 1080),
    }
    assert build_chain(effects, caps) == ["bloom", "tone", "fxaa"]


def test_defect_02_validate_reports_cycle_without_recursion():
    effects = [
        {
            "name": "a",
            "inputs": ["c_out"],
            "outputs": ["a_out"],
            "format": "rgba16f",
        },
        {
            "name": "b",
            "inputs": ["a_out"],
            "outputs": ["b_out"],
            "format": "rgba16f",
        },
        {
            "name": "c",
            "inputs": ["b_out"],
            "outputs": ["c_out"],
            "format": "rgba16f",
        },
    ]
    caps = {
        "max_targets": 4,
        "supported_formats": ("rgba16f", "rgba8"),
        "base_size": (1920, 1080),
    }
    issues = validate_chain(effects, caps)
    assert any(issue.startswith("cycle:") for issue in issues)


def test_defect_03_allocate_targets_reuses_temporary_targets():
    effects = [
        {"name": "fx00", "inputs": ["scene"], "outputs": ["tmp0"], "format": "rgba16f"},
        {"name": "fx01", "inputs": ["tmp0"], "outputs": ["tmp1"], "format": "rgba16f"},
        {"name": "fx02", "inputs": ["tmp1"], "outputs": ["tmp2"], "format": "rgba16f"},
        {"name": "fx03", "inputs": ["tmp2"], "outputs": ["tmp3"], "format": "rgba16f"},
        {"name": "fx04", "inputs": ["tmp3"], "outputs": ["tmp4"], "format": "rgba16f"},
        {"name": "fx05", "inputs": ["tmp4"], "outputs": ["tmp5"], "format": "rgba16f"},
        {"name": "fx06", "inputs": ["tmp5"], "outputs": ["tmp6"], "format": "rgba16f"},
        {"name": "fx07", "inputs": ["tmp6"], "outputs": ["tmp7"], "format": "rgba16f"},
        {"name": "fx08", "inputs": ["tmp7"], "outputs": ["tmp8"], "format": "rgba16f"},
        {"name": "fx09", "inputs": ["tmp8"], "outputs": ["tmp9"], "format": "rgba16f"},
    ]
    result = allocate_targets(effects, ["rt-a", "rt-b"], "rgba16f")
    assert result["count"] == 2
    assert len(set(result["assign"].values())) == 2
    assert set(result["assign"].values()) <= {"rt-a", "rt-b"}
    assert set(result["assign"]) == {effect["name"] for effect in effects}


def test_defect_04_resolve_format_rejects_mid_chain_rgba8():
    effects = [
        {"name": "bloom", "inputs": ["scene"], "outputs": ["hdr"], "format": "rgba16f"},
        {"name": "tone", "inputs": ["hdr"], "outputs": ["graded"], "format": "rgba16f"},
        {"name": "fxaa", "inputs": ["graded"], "outputs": ["aa"], "format": "rgba8"},
        {"name": "present", "inputs": ["aa"], "outputs": ["swapchain"], "format": "rgba8"},
    ]
    with pytest.raises(PostFxError):
        resolve_format(effects, "rgba8")


def test_defect_05_insert_barriers_covers_same_pass_read_write():
    chain = [
        {
            "name": "bloom",
            "inputs": ["hdr", "bloom_tmp"],
            "outputs": ["bloom_tmp"],
            "format": "rgba16f",
        },
        {
            "name": "tone",
            "inputs": ["bloom_tmp"],
            "outputs": ["ldr"],
            "format": "rgba16f",
        },
    ]
    barriers = insert_barriers(chain)
    assert ("bloom", "bloom_tmp", "rw") in barriers
    assert ("tone", "bloom_tmp", "raw") in barriers


def test_defect_06_scale_targets_ceils_odd_half_resolution():
    chain = [
        {
            "name": "bloom",
            "inputs": ["scene"],
            "outputs": ["bloom_rt"],
            "format": "rgba16f",
            "scale": 0.5,
        }
    ]
    sizes = scale_targets(chain, 101, 99, 1.0)
    assert sizes["bloom"] == (51, 50)
    assert sizes["bloom"][0] * 2 >= 101
    assert sizes["bloom"][1] * 2 >= 99


def test_defect_07_merge_passes_rejects_format_mismatch():
    hdr = {
        "name": "tone",
        "inputs": ["hdr"],
        "outputs": ["ldr"],
        "format": "rgba16f",
    }
    ldr = {
        "name": "tone-fxaa",
        "inputs": ["hdr"],
        "outputs": ["ldr"],
        "format": "rgba8",
    }
    same = {
        "name": "tone-alt",
        "inputs": ["hdr"],
        "outputs": ["ldr"],
        "format": "rgba16f",
    }
    assert merge_passes(hdr, ldr) is False
    assert merge_passes(hdr, same) is True


def test_defect_08_validate_chain_reports_out_of_bounds_sampling():
    chain = [
        {
            "name": "bloom",
            "inputs": ["scene"],
            "outputs": ["bloom_rt"],
            "format": "rgba16f",
            "sample_rect": [1900, 1000, 32, 32],
        }
    ]
    caps = {
        "max_targets": 4,
        "supported_formats": ("rgba16f", "rgba8"),
        "base_size": (1920, 1080),
    }
    issues = validate_chain(chain, caps)
    assert any(
        issue.startswith("sample:") and "bloom" in issue for issue in issues
    )


def test_defect_09_run_frames_fails_loudly_when_over_budget():
    chain = [
        {
            "name": "bloom",
            "inputs": ["scene"],
            "outputs": ["hdr_rt"],
            "format": "rgba16f",
        },
        {
            "name": "tone",
            "inputs": ["hdr_rt"],
            "outputs": ["ldr_rt"],
            "format": "rgba8",
        },
    ]
    result = run_frames(["frame-1", "frame-2"], chain, {"max_targets": 1})
    assert result["ok"] is False
    assert result["skipped"] == ["frame-1", "frame-2"]
    assert result["reason"] == "budget"
    assert "degraded" not in result
    assert "pass_through" not in result
    assert result.get("frames", []) == []


def test_defect_10_run_frames_isolates_rt_state_between_frames():
    chain = [
        {
            "name": "bloom",
            "inputs": ["scene"],
            "outputs": ["hdr_rt"],
            "format": "rgba16f",
        },
        {
            "name": "tone",
            "inputs": ["hdr_rt"],
            "outputs": ["ldr_rt"],
            "format": "rgba8",
        },
    ]
    frames = ["frame-1", "frame-2"]
    result = run_frames(frames, chain, {"max_targets": 4})
    assert result["ok"] is True
    assert [record["generation"] for record in result["frames"]] == [0, 1]
    for record in result["frames"]:
        assert set(record["outputs"]) == {"hdr_rt", "ldr_rt"}
        assert set(record["outputs"].values()) == {record["frame"]}


def test_combo_empty_chain_returns_empty_plan():
    caps = {
        "max_targets": 4,
        "supported_formats": ("rgba16f", "rgba8"),
        "base_size": (1920, 1080),
    }
    assert build_chain([], caps) == []
    assert allocate_targets([], ["rt-a"], "rgba16f") == {
        "assign": {},
        "count": 0,
    }
    assert insert_barriers([]) == []
    assert scale_targets([], 1920, 1080, 1.0) == {}
    assert validate_chain([], caps) == []


def test_combo_scale_one_matches_base_dimensions():
    chain = [
        {
            "name": "bloom",
            "inputs": ["scene"],
            "outputs": ["hdr_rt"],
            "format": "rgba16f",
        },
        {
            "name": "fxaa",
            "inputs": ["hdr_rt"],
            "outputs": ["ldr_rt"],
            "format": "rgba8",
        },
    ]
    assert scale_targets(chain, 1279, 721, 1.0) == {
        "bloom": (1279, 721),
        "fxaa": (1279, 721),
    }


def test_combo_cyclic_dependency_raises_postfx_error():
    effects = [
        {
            "name": "a",
            "inputs": ["c_out"],
            "outputs": ["a_out"],
            "format": "rgba16f",
        },
        {
            "name": "b",
            "inputs": ["a_out"],
            "outputs": ["b_out"],
            "format": "rgba16f",
        },
        {
            "name": "c",
            "inputs": ["b_out"],
            "outputs": ["c_out"],
            "format": "rgba16f",
        },
    ]
    caps = {
        "max_targets": 4,
        "supported_formats": ("rgba16f", "rgba8"),
        "base_size": (1920, 1080),
    }
    with pytest.raises(PostFxError, match="cycle"):
        build_chain(effects, caps)


def test_combo_only_final_step_converts_to_rgba8():
    effects = [
        {"name": "bloom", "inputs": ["scene"], "outputs": ["hdr"], "format": "rgba16f"},
        {"name": "tone", "inputs": ["hdr"], "outputs": ["graded"], "format": "rgba16f"},
        {"name": "present", "inputs": ["graded"], "outputs": ["swapchain"], "format": "rgba8"},
    ]
    assert resolve_format(effects, "rgba8") == "rgba8"
    assert resolve_format(effects[:2], "rgba16f") == "rgba16f"
    assert [effect["format"] for effect in effects[:-1]] == [
        "rgba16f",
        "rgba16f",
    ]
    assert effects[-1]["format"] == "rgba8"


def test_combo_rt_count_does_not_grow_with_frame_count():
    chain = [
        {
            "name": "bloom",
            "inputs": ["scene"],
            "outputs": ["hdr_rt"],
            "format": "rgba16f",
        },
        {
            "name": "tone",
            "inputs": ["hdr_rt"],
            "outputs": ["ldr_rt"],
            "format": "rgba8",
        },
    ]
    one_frame = run_frames(["frame-1"], chain, {"max_targets": 4})
    many_frames = run_frames(
        ["frame-1", "frame-2", "frame-3"], chain, {"max_targets": 4}
    )
    assert one_frame["ok"] is True
    assert many_frames["ok"] is True
    assert many_frames["rt_count"] == one_frame["rt_count"]
    assert many_frames["rt_count"] <= 2
    assert len(many_frames["frames"]) == 3
