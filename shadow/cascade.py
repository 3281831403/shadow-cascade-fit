"""Cascaded shadow-map (PSSM) fitting primitives.

All distances, extents and positions are world-space metres.  A camera is a
mapping with exactly these inline fields::

    {"near": 0.1, "far": 100.0, "fov": 1.0471975511965976,
     "aspect": 1.7777777777777777, "pos": (0.0, 0.0, 0.0),
     "dir": (0.0, 0.0, -1.0)}

``fov`` is the vertical full angle in radians.  ``lambda_`` is the practical
split ratio in ``[0.0, 1.0]`` where ``0`` is fully linear and ``1`` is fully
logarithmic.  Texel snapping aligns bounds to multiples of ``texel_size``
metres.  Floating-point assertions use an absolute tolerance of ``1e-6``.
Nothing here uses randomness, wall-clock time, or the network.
"""

from __future__ import annotations

import math

__all__ = [
    "CascadeError",
    "split_scheme",
    "fit_ortho",
    "snap_to_texel",
    "stabilize",
    "light_space_matrix",
    "cull_casters",
    "texel_density",
    "cascade_batch",
]

_TOL = 1e-6


class CascadeError(ValueError):
    """Raised when cascade input is invalid or a batch must be rolled back."""


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _require_number(value, name: str) -> float:
    if not _is_number(value) or not math.isfinite(value):
        raise CascadeError(f"{name} must be a finite number, got {value!r}")
    return float(value)


def _require_vec3(value, name: str) -> tuple[float, float, float]:
    try:
        x, y, z = value
    except (TypeError, ValueError) as exc:
        raise CascadeError(f"{name} must be a 3-component vector") from exc
    return (
        _require_number(x, f"{name}[0]"),
        _require_number(y, f"{name}[1]"),
        _require_number(z, f"{name}[2]"),
    )


def split_scheme(near: float, far: float, count: int, lambda_: float) -> list[float]:
    """Return the far plane of every cascade, in metres.

    ``near`` and ``far`` delimit the camera range and must satisfy
    ``0 < near < far``.  ``count`` is the number of cascades and must be at
    least ``1``.  ``lambda_`` is the practical split ratio in ``[0, 1]``:
    ``0`` selects purely linear splits, ``1`` selects purely logarithmic
    splits, and intermediate values interpolate between them.

    For ``i = 1..count`` with ``p = i / count`` the split is::

        lambda_ * near * (far / near) ** p
            + (1 - lambda_) * (near + (far - near) * p)

    The result has ``count`` entries, is strictly increasing, and its last
    entry equals ``far`` exactly.  Invalid distances, ``count < 1`` or a
    ratio outside ``[0, 1]`` raise :class:`CascadeError`.  Results are
    compared with an absolute tolerance of ``1e-6``.
    """

    near = _require_number(near, "near")
    far = _require_number(far, "far")
    lambda_ = _require_number(lambda_, "lambda_")
    if isinstance(count, bool) or not isinstance(count, int):
        raise CascadeError(f"count must be an integer, got {count!r}")
    if near <= 0.0:
        raise CascadeError(f"near must be positive, got {near}")
    if far <= near:
        raise CascadeError(f"far must exceed near, got far={far}, near={near}")
    if count < 1:
        raise CascadeError(f"count must be at least 1, got {count}")
    if not 0.0 <= lambda_ <= 1.0:
        raise CascadeError(f"lambda_ must be in [0, 1], got {lambda_}")

    splits = []
    for i in range(1, count + 1):
        p = i / count
        log_split = near * (far / near) ** p
        linear_split = near + (far - near) * p
        splits.append(lambda_ * log_split + (1.0 - lambda_) * linear_split)
    return splits


def fit_ortho(
    split_near: float, split_far: float, fov: float, aspect: float
) -> dict:
    """Fit the orthographic extents of one cascade.

    The cascade covers the camera slice from ``split_near`` to ``split_far``
    metres, with ``0 < split_near < split_far``.  ``fov`` is the vertical
    full field of view in radians and ``aspect`` is the positive
    width/height ratio of the viewport.  The far end of a symmetric
    perspective slice dominates its extent, so::

        half_height = split_far * tan(fov / 2)
        height = 2 * half_height
        width = height * aspect

    Return ``{"width": width, "height": height}`` in metres.  This expansion
    by ``fov`` and ``aspect`` keeps the wide sides of the slice inside the
    orthographic box.  Invalid values raise :class:`CascadeError`; extents
    are compared with an absolute tolerance of ``1e-6``.
    """

    split_near = _require_number(split_near, "split_near")
    split_far = _require_number(split_far, "split_far")
    fov = _require_number(fov, "fov")
    aspect = _require_number(aspect, "aspect")
    if split_near <= 0.0:
        raise CascadeError(f"split_near must be positive, got {split_near}")
    if split_far <= split_near:
        raise CascadeError(
            f"split_far must exceed split_near, got {split_far} <= {split_near}"
        )
    if not 0.0 < fov < math.pi:
        raise CascadeError(f"fov must be in (0, pi), got {fov}")
    if aspect <= 0.0:
        raise CascadeError(f"aspect must be positive, got {aspect}")

    height = 2.0 * split_far * math.tan(fov / 2.0)
    width = height * aspect
    return {"width": width, "height": height}


def snap_to_texel(bounds: dict, texel_size: float) -> dict:
    """Snap a world-space XY bounding box outward to a texel grid.

    ``bounds`` is a mapping with numeric ``min_x``, ``max_x``, ``min_y`` and
    ``max_y`` values in metres, where each minimum is at most its maximum.
    ``texel_size`` is the positive size of one shadow-map texel in metres.

    Each edge is moved outward to the nearest grid line::

        min_x = floor(min_x / texel_size) * texel_size
        max_x = ceil(max_x / texel_size) * texel_size
        min_y = floor(min_y / texel_size) * texel_size
        max_y = ceil(max_y / texel_size) * texel_size

    Return a new mapping with those four keys; the snapped box covers the
    original box and every edge is an exact multiple of ``texel_size`` within
    the ``1e-6`` tolerance.  This removes sub-texel movement that would make
    shadow edges shimmer as the camera moves.  Invalid input raises
    :class:`CascadeError`.
    """

    if not isinstance(bounds, dict):
        raise CascadeError("bounds must be a mapping")
    texel_size = _require_number(texel_size, "texel_size")
    if texel_size <= 0.0:
        raise CascadeError(f"texel_size must be positive, got {texel_size}")

    values = {}
    for key in ("min_x", "max_x", "min_y", "max_y"):
        if key not in bounds:
            raise CascadeError(f"bounds is missing {key!r}")
        values[key] = _require_number(bounds[key], key)
    if values["min_x"] > values["max_x"] or values["min_y"] > values["max_y"]:
        raise CascadeError("bounds minimum exceeds its maximum")

    return {
        "min_x": math.floor(values["min_x"] / texel_size) * texel_size,
        "max_x": math.ceil(values["max_x"] / texel_size) * texel_size,
        "min_y": math.floor(values["min_y"] / texel_size) * texel_size,
        "max_y": math.ceil(values["max_y"] / texel_size) * texel_size,
    }


def stabilize(prev: dict, cur: dict, ratio: float) -> dict:
    """Blend the previous and current cascade bounds during a transition.

    ``prev`` and ``cur`` are mappings with the same numeric keys, typically
    the bounding boxes returned by :func:`snap_to_texel`, with values in
    metres.  ``ratio`` is in ``[0, 1]`` and weights the current bounds::

        result[key] = prev[key] + ratio * (cur[key] - prev[key])

    Thus ``ratio == 0`` keeps ``prev``, ``ratio == 1`` keeps ``cur``, and
    intermediate values damp the jump when control moves between cascades,
    avoiding a bright seam.  Return a new mapping; neither input is
    modified.  Mismatched keys or an invalid ratio raise
    :class:`CascadeError`.  Results are compared with an absolute tolerance
    of ``1e-6``.
    """

    if not isinstance(prev, dict) or not isinstance(cur, dict):
        raise CascadeError("prev and cur must be mappings")
    ratio = _require_number(ratio, "ratio")
    if not 0.0 <= ratio <= 1.0:
        raise CascadeError(f"ratio must be in [0, 1], got {ratio}")
    if set(prev.keys()) != set(cur.keys()):
        raise CascadeError("prev and cur must have the same keys")

    blended = {}
    for key in prev:
        before = _require_number(prev[key], f"prev[{key!r}]")
        after = _require_number(cur[key], f"cur[{key!r}]")
        blended[key] = before + ratio * (after - before)
    return blended


def light_space_matrix(bounds: dict, light_dir: tuple[float, float, float]) -> list[list[float]]:
    """Return a row-major 4x4 world-to-light orthographic matrix.

    ``bounds`` is a world-space axis-aligned box in metres with ``min_x``,
    ``max_x``, ``min_y``, ``max_y``, ``min_z`` and ``max_z`` keys; every
    minimum must be strictly below its maximum.  ``light_dir`` is a non-zero
    light propagation direction in world space and is normalized internally.

    The light basis is right-handed::

        forward = normalize(light_dir)
        right   = normalize(cross(up, forward))
        up      = cross(forward, right)

    World up ``(0, 1, 0)`` is used unless the light is vertical, in which case
    ``(1, 0, 0)`` is used.  The eight box corners are transformed into that
    basis, the resulting light-space AABB is mapped to ``[-1, 1]^3`` by an
    orthographic projection, and the returned matrix is ``projection @
    view``.  Its upper-left 3x3 block must be a positively scaled orthonormal
    basis (pairwise row dot products are zero within ``1e-6``), its last row
    must be ``[0, 0, 0, 1]``, and its determinant must be non-zero so the
    orthographic projection does not collapse or shear the shadow map.

    Invalid bounds or a zero direction raises :class:`CascadeError`.
    """

    if not isinstance(bounds, dict):
        raise CascadeError("bounds must be a mapping")
    keys = ("min_x", "max_x", "min_y", "max_y", "min_z", "max_z")
    values = {}
    for key in keys:
        if key not in bounds:
            raise CascadeError(f"bounds is missing {key!r}")
        values[key] = _require_number(bounds[key], key)
    for axis in ("x", "y", "z"):
        if not values[f"min_{axis}"] < values[f"max_{axis}"]:
            raise CascadeError(
                f"bounds min_{axis} must be strictly below max_{axis}"
            )

    forward = _require_vec3(light_dir, "light_dir")
    length = math.sqrt(sum(component * component for component in forward))
    if length <= 0.0:
        raise CascadeError("light_dir must be non-zero")
    forward = tuple(component / length for component in forward)

    world_up = (0.0, 1.0, 0.0)
    if abs(sum(a * b for a, b in zip(forward, world_up))) > 1.0 - 1e-9:
        world_up = (1.0, 0.0, 0.0)

    def cross(a, b):
        return (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        )

    right = cross(world_up, forward)
    right_length = math.sqrt(sum(c * c for c in right))
    right = tuple(c / right_length for c in right)
    up = cross(forward, right)

    # View matrix rows are the basis vectors; translation follows.
    view = [
        [right[0], right[1], right[2], 0.0],
        [up[0], up[1], up[2], 0.0],
        [forward[0], forward[1], forward[2], 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]

    corners = [
        (values[f"{edge_x}_x"], values[f"{edge_y}_y"], values[f"{edge_z}_z"])
        for edge_x in ("min", "max")
        for edge_y in ("min", "max")
        for edge_z in ("min", "max")
    ]
    light_corners = [
        (
            sum(view[0][k] * corner[k] for k in range(3)),
            sum(view[1][k] * corner[k] for k in range(3)),
            sum(view[2][k] * corner[k] for k in range(3)),
        )
        for corner in corners
    ]
    lo = [min(corner[k] for corner in light_corners) for k in range(3)]
    hi = [max(corner[k] for corner in light_corners) for k in range(3)]

    scale = [2.0 / (hi[k] - lo[k]) if hi[k] > lo[k] else 1.0 for k in range(3)]
    offset = [-(hi[k] + lo[k]) / 2.0 * scale[k] for k in range(3)]

    # projection @ view, row-major.
    matrix = []
    for row in range(3):
        matrix.append(
            [scale[row] * view[row][k] for k in range(3)] + [offset[row]]
        )
    matrix.append([0.0, 0.0, 0.0, 1.0])
    return matrix


def cull_casters(objects: list[dict], cascade: dict) -> list[int]:
    """Return the ids of objects that can cast into one cascade.

    ``cascade`` is ``{"near": near, "far": far, "pos": (x, y, z),
    "dir": (x, y, z)}`` with distances in metres and a non-zero direction.
    ``objects`` is a sequence of ``{"id": int, "center": (x, y, z),
    "radius": radius}`` mappings, where ``radius >= 0`` is the object's
    bounding-sphere radius in metres.

    An object's centre depth along the cascade view direction is::

        d = dot(center - pos, normalize(dir))

    The object is kept when its sphere interval ``[d - radius, d + radius]``
    intersects ``[near, far]`` within ``1e-6``.  Testing the whole interval,
    rather than only the centre, keeps casters that straddle a cascade
    boundary.  Return ids in input order; invalid input raises
    :class:`CascadeError`.
    """

    if not isinstance(cascade, dict):
        raise CascadeError("cascade must be a mapping")
    for key in ("near", "far", "pos", "dir"):
        if key not in cascade:
            raise CascadeError(f"cascade is missing {key!r}")
    near = _require_number(cascade["near"], "cascade['near']")
    far = _require_number(cascade["far"], "cascade['far']")
    if near < 0.0:
        raise CascadeError(f"cascade near must be non-negative, got {near}")
    if far < near:
        raise CascadeError(f"cascade far must be >= near, got {far} < {near}")
    pos = _require_vec3(cascade["pos"], "cascade['pos']")
    direction = _require_vec3(cascade["dir"], "cascade['dir']")
    length = math.sqrt(sum(c * c for c in direction))
    if length <= 0.0:
        raise CascadeError("cascade dir must be non-zero")
    direction = tuple(c / length for c in direction)

    if not isinstance(objects, (list, tuple)):
        raise CascadeError("objects must be a sequence")

    kept = []
    for obj in objects:
        if not isinstance(obj, dict):
            raise CascadeError("each object must be a mapping")
        for key in ("id", "center", "radius"):
            if key not in obj:
                raise CascadeError(f"object is missing {key!r}")
        if isinstance(obj["id"], bool) or not isinstance(obj["id"], int):
            raise CascadeError(f"object id must be an int, got {obj['id']!r}")
        center = _require_vec3(obj["center"], "object center")
        radius = _require_number(obj["radius"], "object radius")
        if radius < 0.0:
            raise CascadeError(f"object radius must be >= 0, got {radius}")

        depth = sum(
            (center[k] - pos[k]) * direction[k] for k in range(3)
        )
        if depth + radius >= near - _TOL and depth - radius <= far + _TOL:
            kept.append(obj["id"])
    return kept


def texel_density(cascade: dict, resolution: int) -> float:
    """Return the cascade's worst-axis texel density in texels per metre.

    ``cascade`` is ``{"width": width, "height": height}`` with positive
    orthographic extents in metres.  ``resolution`` is the positive edge
    length of the square shadow map in texels.  Because one texel must cover
    both axes, the conservative density is::

        resolution / max(width, height)

    Four PSSM cascades fitted with resolutions proportional to their extents
    therefore have densities within 10% of one another.  Invalid values raise
    :class:`CascadeError`; the result is compared with an absolute tolerance
    of ``1e-6``.
    """

    if not isinstance(cascade, dict):
        raise CascadeError("cascade must be a mapping")
    for key in ("width", "height"):
        if key not in cascade:
            raise CascadeError(f"cascade is missing {key!r}")
    width = _require_number(cascade["width"], "width")
    height = _require_number(cascade["height"], "height")
    if width <= 0.0 or height <= 0.0:
        raise CascadeError("cascade extents must be positive")
    if isinstance(resolution, bool) or not isinstance(resolution, int):
        raise CascadeError(f"resolution must be an integer, got {resolution!r}")
    if resolution <= 0:
        raise CascadeError(f"resolution must be positive, got {resolution}")

    return resolution / max(width, height)


def _validate_camera(camera, index: int) -> None:
    if not isinstance(camera, dict):
        raise CascadeError(f"camera {index} must be a mapping")
    for key in ("near", "far", "fov", "aspect", "pos", "dir"):
        if key not in camera:
            raise CascadeError(f"camera {index} is missing {key!r}")
    near = _require_number(camera["near"], f"camera {index} near")
    far = _require_number(camera["far"], f"camera {index} far")
    fov = _require_number(camera["fov"], f"camera {index} fov")
    aspect = _require_number(camera["aspect"], f"camera {index} aspect")
    if near <= 0.0:
        raise CascadeError(f"camera {index} near must be positive, got {near}")
    if far <= near:
        raise CascadeError(
            f"camera {index} far must exceed near, got {far} <= {near}"
        )
    if not 0.0 < fov < math.pi:
        raise CascadeError(f"camera {index} fov must be in (0, pi), got {fov}")
    if aspect <= 0.0:
        raise CascadeError(f"camera {index} aspect must be positive, got {aspect}")
    _require_vec3(camera["pos"], f"camera {index} pos")
    direction = _require_vec3(camera["dir"], f"camera {index} dir")
    if all(component == 0.0 for component in direction):
        raise CascadeError(f"camera {index} dir must be non-zero")


def cascade_batch(cameras: list[dict], config: dict) -> dict:
    """Fit a list of inline cameras with per-camera isolation.

    Each camera has the inline keys ``near``, ``far``, ``fov``, ``aspect``,
    ``pos`` and ``dir``, as described in the module docstring.  ``config``
    has ``count`` (at least 1) and ``lambda_`` (in ``[0, 1]``).

    Return::

        {"cameras": [
            {"index": i,
             "near": camera["near"], "far": camera["far"],
             "fov": camera["fov"], "aspect": camera["aspect"],
             "splits": [...],
             "ortho": [{"width": ..., "height": ...}, ...]},
            ...
        ]}

    Entries preserve input order.  Every camera gets its own split and ortho
    buffers; no bounds or partial result is shared between cameras.

    Validation is all-or-nothing: every camera and the config are checked
    before fitting starts.  If any camera is invalid (including a
    non-positive near/far/fov/aspect or a zero direction), or if
    ``count < 1``, raise :class:`CascadeError`.  No partial result is
    returned and neither ``cameras`` nor ``config`` is mutated, so a failed
    batch leaves the caller's state untouched.  Results use an absolute
    tolerance of ``1e-6``.
    """

    if not isinstance(cameras, (list, tuple)):
        raise CascadeError("cameras must be a sequence")
    if not isinstance(config, dict):
        raise CascadeError("config must be a mapping")
    for key in ("count", "lambda_"):
        if key not in config:
            raise CascadeError(f"config is missing {key!r}")
    count = config["count"]
    if isinstance(count, bool) or not isinstance(count, int):
        raise CascadeError(f"count must be an integer, got {count!r}")
    if count < 1:
        raise CascadeError(f"count must be at least 1, got {count}")
    lambda_ = _require_number(config["lambda_"], "lambda_")
    if not 0.0 <= lambda_ <= 1.0:
        raise CascadeError(f"lambda_ must be in [0, 1], got {lambda_}")

    # Validate everything up front so a failure rolls back the whole batch.
    for index, camera in enumerate(cameras):
        _validate_camera(camera, index)

    entries = []
    for index, camera in enumerate(cameras):
        near = float(camera["near"])
        far = float(camera["far"])
        fov = float(camera["fov"])
        aspect = float(camera["aspect"])
        splits = split_scheme(near, far, count, lambda_)
        ortho = []
        slice_near = near
        for split_far in splits:
            ortho.append(fit_ortho(slice_near, split_far, fov, aspect))
            slice_near = split_far
        entries.append(
            {
                "index": index,
                "near": near,
                "far": far,
                "fov": fov,
                "aspect": aspect,
                "splits": splits,
                "ortho": ortho,
            }
        )
    return {"cameras": entries}
