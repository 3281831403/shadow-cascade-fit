"""Acceptance tests for the shadow-cascade fitting bugfix task.

All geometry is in world-space metres.  Cameras are plain inline mappings with
``near``, ``far``, ``fov`` (radians), ``aspect``, ``pos`` and ``dir``.  These
tests are deterministic: they use no randomness, sleep, or network.
"""

from __future__ import annotations

import math

import pytest

from shadow.cascade import (
    CascadeError,
    cascade_batch,
    cull_casters,
    fit_ortho,
    light_space_matrix,
    snap_to_texel,
    split_scheme,
    stabilize,
    texel_density,
)


def _det4(matrix: list[list[float]]) -> float:
    """Return the determinant of a 4x4 row-major matrix."""

    if len(matrix) == 1:
        return matrix[0][0]
    total = 0.0
    for column, value in enumerate(matrix[0]):
        minor = [row[:column] + row[column + 1 :] for row in matrix[1:]]
        total += (-1.0) ** column * value * _det4(minor)
    return total


def test_split_scheme_weights_near_cascades_more_than_linear():
    near, far, count = 0.1, 100.0, 4
    splits = split_scheme(near, far, count, 0.75)
    linear_first = near + (far - near) / count

    assert splits[0] < linear_first
    assert splits[0] > near


def test_split_scheme_lambda_one_is_logarithmic_and_ends_at_far():
    near, far, count = 0.5, 80.0, 5
    splits = split_scheme(near, far, count, 1.0)

    for index, split in enumerate(splits, start=1):
        expected = near * (far / near) ** (index / count)
        assert split == pytest.approx(expected, abs=1e-6)
    assert splits[-1] == pytest.approx(far, abs=1e-6)


def test_fit_ortho_expands_horizontally_by_aspect_and_fov():
    fov = math.radians(60.0)
    aspect = 16.0 / 9.0
    fitted = fit_ortho(0.5, 20.0, fov, aspect)
    expected_height = 2.0 * 20.0 * math.tan(fov / 2.0)

    assert fitted["height"] == pytest.approx(expected_height, abs=1e-6)
    assert fitted["width"] == pytest.approx(expected_height * aspect, abs=1e-6)
    assert fitted["width"] > fitted["height"]

    narrow = fit_ortho(0.5, 20.0, math.radians(30.0), aspect)
    assert narrow["height"] < fitted["height"]


def test_snap_to_texel_quantizes_bounds_without_losing_coverage():
    original = {"min_x": -1.3, "max_x": 2.4, "min_y": 0.2, "max_y": 3.1}
    snapped = snap_to_texel(original, 0.5)

    assert snapped["min_x"] == pytest.approx(-1.5, abs=1e-6)
    assert snapped["max_x"] == pytest.approx(2.5, abs=1e-6)
    assert snapped["min_y"] == pytest.approx(0.0, abs=1e-6)
    assert snapped["max_y"] == pytest.approx(3.5, abs=1e-6)

    for axis in ("x", "y"):
        for edge in ("min", "max"):
            value = snapped[f"{edge}_{axis}"]
            assert abs(value / 0.5 - round(value / 0.5)) <= 1e-6
    assert snapped["min_x"] <= original["min_x"]
    assert snapped["max_x"] >= original["max_x"]
    assert snapped["min_y"] <= original["min_y"]
    assert snapped["max_y"] >= original["max_y"]


def test_stabilize_blends_previous_and_current_by_ratio():
    previous = {"min_x": -4.0, "max_x": 4.0, "min_y": -2.0, "max_y": 2.0}
    current = {"min_x": -2.0, "max_x": 6.0, "min_y": -1.0, "max_y": 3.0}
    blended = stabilize(previous, current, 0.25)

    for key in previous:
        expected = previous[key] + 0.25 * (current[key] - previous[key])
        assert blended[key] == pytest.approx(expected, abs=1e-6)
    assert stabilize(previous, current, 0.0)["min_x"] == pytest.approx(-4.0, abs=1e-6)
    assert stabilize(previous, current, 1.0)["max_x"] == pytest.approx(6.0, abs=1e-6)


def test_light_space_matrix_has_orthonormal_basis_and_nonzero_determinant():
    bounds = {
        "min_x": -4.0,
        "max_x": 4.0,
        "min_y": -2.0,
        "max_y": 3.0,
        "min_z": -1.0,
        "max_z": 1.0,
    }
    matrix = light_space_matrix(bounds, (0.3, -1.0, 0.2))

    assert len(matrix) == 4
    assert all(len(row) == 4 for row in matrix)
    assert matrix[3] == pytest.approx([0.0, 0.0, 0.0, 1.0], abs=1e-6)

    rows = [row[:3] for row in matrix[:3]]
    for left in range(3):
        for right in range(left + 1, 3):
            dot = sum(rows[left][axis] * rows[right][axis] for axis in range(3))
            assert dot == pytest.approx(0.0, abs=1e-6)
    assert abs(_det4(matrix)) > 1e-9


def test_cull_casters_keeps_cross_cascade_objects():
    cascade = {
        "near": 2.0,
        "far": 10.0,
        "pos": (0.0, 0.0, 0.0),
        "dir": (0.0, 0.0, -1.0),
    }
    objects = [
        {"id": 1, "center": (0.0, 0.0, -5.0), "radius": 0.5},
        {"id": 2, "center": (0.0, 0.0, -11.0), "radius": 2.0},
        {"id": 3, "center": (0.0, 0.0, -13.0), "radius": 1.0},
        {"id": 4, "center": (0.0, 0.0, -1.0), "radius": 1.5},
    ]

    assert cull_casters(objects, cascade) == [1, 2, 4]


def test_texel_density_is_resolution_based_and_uniform():
    cascade = {"width": 16.0, "height": 8.0}
    low = texel_density(cascade, 800)
    high = texel_density(cascade, 1600)

    assert low == pytest.approx(50.0, abs=1e-6)
    assert high == pytest.approx(100.0, abs=1e-6)
    assert high == pytest.approx(2.0 * low, abs=1e-6)

    cascades = [
        {"width": 8.0 * 2**index, "height": 4.0 * 2**index}
        for index in range(4)
    ]
    resolutions = [800 * 2**index for index in range(4)]
    densities = [
        texel_density(item, resolution)
        for item, resolution in zip(cascades, resolutions)
    ]
    baseline = densities[0]
    assert all(abs(value - baseline) / baseline <= 0.10 for value in densities)


def test_cascade_batch_keeps_per_camera_ortho_bounds_isolated():
    cameras = [
        {
            "near": 0.1,
            "far": 10.0,
            "fov": math.radians(60.0),
            "aspect": 1.0,
            "pos": (0.0, 0.0, 0.0),
            "dir": (0.0, 0.0, -1.0),
        },
        {
            "near": 0.1,
            "far": 10.0,
            "fov": math.radians(30.0),
            "aspect": 2.0,
            "pos": (4.0, 0.0, 0.0),
            "dir": (0.0, 0.0, -1.0),
        },
    ]
    result = cascade_batch(cameras, {"count": 2, "lambda_": 0.5})
    first, second = result["cameras"]

    assert first["ortho"][0]["width"] == pytest.approx(
        first["ortho"][0]["height"], abs=1e-6
    )
    assert second["ortho"][0]["width"] == pytest.approx(
        2.0 * second["ortho"][0]["height"], abs=1e-6
    )
    assert second["ortho"][0]["height"] < first["ortho"][0]["height"]
    assert first["splits"] == second["splits"]
    assert first["ortho"] != second["ortho"]


def test_batch_rejects_zero_count_before_any_output():
    cameras = [
        {
            "near": 0.1,
            "far": 10.0,
            "fov": math.radians(60.0),
            "aspect": 1.0,
            "pos": (0.0, 0.0, 0.0),
            "dir": (0.0, 0.0, -1.0),
        }
    ]

    with pytest.raises(CascadeError):
        cascade_batch(cameras, {"count": 0, "lambda_": 0.5})


def test_split_scheme_count_one_returns_only_far():
    splits = split_scheme(0.25, 75.0, 1, 0.6)

    assert len(splits) == 1
    assert splits[0] == pytest.approx(75.0, abs=1e-6)


def test_split_scheme_lambda_zero_matches_linear_splits():
    near, far, count = 1.0, 101.0, 5
    splits = split_scheme(near, far, count, 0.0)

    expected = [near + (far - near) * index / count for index in range(1, count + 1)]
    assert splits == pytest.approx(expected, abs=1e-6)


def test_split_scheme_count_zero_raises_cascade_error():
    with pytest.raises(CascadeError):
        split_scheme(0.1, 100.0, 0, 0.5)


def test_split_scheme_far_planes_strictly_increase():
    splits = split_scheme(0.1, 200.0, 6, 0.6)

    for previous, current in zip(splits, splits[1:]):
        assert current > previous
    assert splits[-1] == pytest.approx(200.0, abs=1e-6)


def test_batch_success_preserves_camera_order():
    cameras = [
        {
            "near": 0.2,
            "far": 50.0,
            "fov": math.radians(50.0),
            "aspect": 1.0,
            "pos": (0.0, 0.0, 0.0),
            "dir": (0.0, 0.0, -1.0),
        },
        {
            "near": 0.3,
            "far": 80.0,
            "fov": math.radians(60.0),
            "aspect": 1.5,
            "pos": (5.0, 1.0, 0.0),
            "dir": (0.0, 0.0, -1.0),
        },
        {
            "near": 0.4,
            "far": 120.0,
            "fov": math.radians(70.0),
            "aspect": 2.0,
            "pos": (-3.0, 2.0, 4.0),
            "dir": (0.0, 0.0, -1.0),
        },
    ]
    result = cascade_batch(cameras, {"count": 3, "lambda_": 0.7})
    entries = result["cameras"]

    assert [entry["index"] for entry in entries] == [0, 1, 2]
    for camera, entry in zip(cameras, entries):
        assert entry["near"] == pytest.approx(camera["near"], abs=1e-6)
        assert entry["far"] == pytest.approx(camera["far"], abs=1e-6)
        assert len(entry["splits"]) == 3
        assert entry["splits"][-1] == pytest.approx(camera["far"], abs=1e-6)
