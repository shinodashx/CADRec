import argparse
import gc
import json
import math
import random
import re
import shutil
import sys
from copy import deepcopy
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_EVEN, getcontext
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from cadlib.curves import Arc, Circle, Line
from cadlib.extrude import CADSequence, Extrude
from cadlib.macro import EXTRUDE_OPERATIONS, EXTENT_TYPE

CANONICAL_PLANES = {
    "XY": {
        "normal": np.array([0.0, 0.0, 1.0]),
        "x_axis": np.array([1.0, 0.0, 0.0]),
        "y_axis": np.array([0.0, 1.0, 0.0]),
    },
    "XZ": {
        "normal": np.array([0.0, -1.0, 0.0]),
        "x_axis": np.array([1.0, 0.0, 0.0]),
        "y_axis": np.array([0.0, 0.0, 1.0]),
    },
    "YZ": {
        "normal": np.array([1.0, 0.0, 0.0]),
        "x_axis": np.array([0.0, 1.0, 0.0]),
        "y_axis": np.array([ 0.0, 0.0, 1.0]),
    },
}

DECIMAL_PLACES = 2
getcontext().prec = 28
_DECIMAL_STEP = Decimal("1").scaleb(-DECIMAL_PLACES)
_ZERO_QUANTIZED = Decimal("0").quantize(_DECIMAL_STEP)

MAX_FREE_STEPS_PER_PART = 3
FACE_GAP_TOLERANCE = 5e-1
FACE_MIN_OVERLAP = 1e-6
SHAPE_CONTACT_TOLERANCE = 1e-5
SHAPE_NEAR_TOLERANCE = FACE_GAP_TOLERANCE
STAR_MIN_SATELLITES = 8
STAR_MAX_RADIUS_CV = 0.35
STAR_MIN_ANGLE_COVERAGE = math.radians(270.0)
STAR_MAX_SATELLITE_VOLUME_CV = 0.75
STAR_MAX_ROUND_HUB_ASPECT = 1.35
RADIAL_AXIAL_OVERLAP_RATIO = 0.05
FULL_PRECISION_SIG_DIGITS = 17
ROUND_CUT_CLEARANCE = 2.0 * 10 ** (-DECIMAL_PLACES)
FloatFormatter = Callable[[float], str]
_FPS_RUNTIME = None
_SHAPE_DISTANCE_CACHE: dict[tuple[int, int], Optional[float]] = {}


def _default_cut_clearance(formatter: FloatFormatter, direct_solid: bool) -> float:
    return ROUND_CUT_CLEARANCE if direct_solid and formatter is _fmt else 0.0
def _quantize_decimal(value: float) -> Decimal:
    if math.isclose(value, 0.0, abs_tol=1e-12):
        value = 0.0
    decimal_value = Decimal(str(value))
    quantized = decimal_value.quantize(_DECIMAL_STEP, rounding=ROUND_HALF_EVEN)
    if quantized == 0:
        return _ZERO_QUANTIZED
    return quantized


def _normalize_to_random_range(cad_seq, scale: float = 1.0) -> None:
    """Normalize CAD sequence to a random symmetric range [-x, x] where x is integer between 75 and 100."""
    target = random.randint(100, 100)
    bbox = cad_seq.bbox  # [max_point, min_point]
    center = (bbox[0] + bbox[1]) / 2
    half_extent = (bbox[0] - bbox[1]) / 2
    max_abs = np.max(np.abs(half_extent))
    if max_abs < 1e-12:
        return
    cad_seq.transform(-center, target / max_abs * scale)


def _normalization_transform_from_minmax_bbox(
    bbox: np.ndarray,
    scale: float = 1.0,
    target: float = 100.0,
) -> tuple[np.ndarray, float]:
    bbox = np.asarray(bbox, dtype=float)
    center = (bbox[0] + bbox[1]) / 2.0
    half_extent = (bbox[1] - bbox[0]) / 2.0
    max_abs = float(np.max(np.abs(half_extent)))
    if max_abs < 1e-12:
        return -center, float(scale)
    return -center, float(target) / max_abs * float(scale)


def _apply_translation_scale(points: np.ndarray, translation: Sequence[float], scale_factor: float) -> np.ndarray:
    transformed = np.asarray(points, dtype=np.float32)
    transformed = transformed + np.asarray(translation, dtype=np.float32)
    transformed = transformed * float(scale_factor)
    return transformed.astype(np.float32)


def _merge_close_values(values: list[float], tol: float = 1e-9) -> list[float]:
    """Merge near-equal float values and return sorted unique list."""
    if not values:
        return []
    values = sorted(values)
    merged = [values[0]]
    for val in values[1:]:
        if not math.isclose(val, merged[-1], abs_tol=tol):
            merged.append(val)
    return merged


def _detect_axis_aligned_rectangle(loop, transform, offset, tol: float = 1e-9):
    """Return (center_x, center_y, width, height) if loop is axis-aligned rectangle."""
    if len(loop.children) != 4:
        return None
    coords: list[tuple[float, float]] = []
    for curve in loop.children:
        if not isinstance(curve, Line):
            return None
        sx, sy = _apply_2d_transform(curve.start_point, transform, offset)
        ex, ey = _apply_2d_transform(curve.end_point, transform, offset)
        dx = ex - sx
        dy = ey - sy
        if not (math.isclose(dx, 0.0, abs_tol=tol) or math.isclose(dy, 0.0, abs_tol=tol)):
            return None
        coords.append((sx, sy))
        coords.append((ex, ey))

    xs = _merge_close_values([pt[0] for pt in coords], tol)
    ys = _merge_close_values([pt[1] for pt in coords], tol)
    if len(xs) != 2 or len(ys) != 2:
        return None
    width = abs(xs[1] - xs[0])
    height = abs(ys[1] - ys[0])
    if width < tol or height < tol:
        return None
    center = ((xs[0] + xs[1]) / 2.0, (ys[0] + ys[1]) / 2.0)
    return center[0], center[1], width, height


def _describe_simple_loop(
    loop,
    transform: Optional[np.ndarray],
    offset: Optional[Sequence[float]],
) -> Optional[tuple[str, tuple[float, float], tuple[float, ...]]]:
    """Return (shape_type, center, dims) for axis-aligned rectangles or circles."""
    if len(loop.children) == 1 and isinstance(loop.children[0], Circle):
        circle = loop.children[0]
        cx, cy = _apply_2d_transform(circle.center, transform, offset)
        r = abs(float(circle.radius))
        return "circle", (cx, cy), (r,)

    rect = _detect_axis_aligned_rectangle(loop, transform, offset)
    if rect:
        cx, cy, width, height = rect
        return "rect", (cx, cy), (width, height)

    return None


def _get_extrude_bbox(extrude: Extrude) -> np.ndarray:
    """Analytic bbox for one CADSequence extrude leaf, matching the root grouping script."""
    profile = deepcopy(extrude.profile)
    profile.denormalize(extrude.sketch_size)

    points = []
    for loop in profile.children:
        for curve in loop.children:
            if isinstance(curve, Line):
                points.append(curve.start_point)
                points.append(curve.end_point)
            elif isinstance(curve, Arc):
                points.append(curve.start_point)
                points.append(curve.mid_point)
                points.append(curve.end_point)
            elif isinstance(curve, Circle):
                cx, cy = curve.center
                r = abs(float(curve.radius))
                points.extend([(cx - r, cy - r), (cx + r, cy + r)])

    if not points:
        return np.zeros((2, 3), dtype=float)

    origin = np.asarray(extrude.sketch_pos, dtype=float)
    x_axis = np.asarray(extrude.sketch_plane.x_axis, dtype=float)
    y_axis = np.asarray(extrude.sketch_plane.y_axis, dtype=float)
    normal = np.asarray(extrude.sketch_plane.normal, dtype=float)

    points_3d = [
        origin + float(pt[0]) * x_axis + float(pt[1]) * y_axis
        for pt in points
    ]

    if extrude.extent_type == EXTENT_TYPE.index("SymmetricFeatureExtentType"):
        extents = [extrude.extent_one, -extrude.extent_one]
    elif extrude.extent_type == EXTENT_TYPE.index("TwoSidesFeatureExtentType"):
        extents = [extrude.extent_one, -extrude.extent_two]
    else:
        extents = [extrude.extent_one, 0.0]

    extended_points = []
    for pt in points_3d:
        extended_points.append(pt)
        for extent in extents:
            extended_points.append(pt + normal * float(extent))

    extended_points = np.asarray(extended_points, dtype=float)
    return np.stack([extended_points.min(axis=0), extended_points.max(axis=0)], axis=0)


def _bbox_array_to_list(bbox: np.ndarray) -> list[float]:
    bbox = np.asarray(bbox, dtype=float)
    return [
        round(float(bbox[0][0]), 2),
        round(float(bbox[0][1]), 2),
        round(float(bbox[0][2]), 2),
        round(float(bbox[1][0]), 2),
        round(float(bbox[1][1]), 2),
        round(float(bbox[1][2]), 2),
    ]


def _bbox_list_to_array(bbox: Sequence[float]) -> np.ndarray:
    return np.array(
        [
            [float(bbox[0]), float(bbox[1]), float(bbox[2])],
            [float(bbox[3]), float(bbox[4]), float(bbox[5])],
        ],
        dtype=float,
    )


def _merge_bbox_arrays(bboxes: Sequence[np.ndarray]) -> np.ndarray:
    bbox_list = [np.asarray(bbox, dtype=float) for bbox in bboxes if bbox is not None]
    if not bbox_list:
        return np.zeros((2, 3), dtype=float)
    mins = np.stack([bbox[0] for bbox in bbox_list], axis=0).min(axis=0)
    maxs = np.stack([bbox[1] for bbox in bbox_list], axis=0).max(axis=0)
    return np.stack([mins, maxs], axis=0)


def _bbox_volume(bbox: np.ndarray) -> float:
    bbox = np.asarray(bbox, dtype=float)
    extents = np.maximum(bbox[1] - bbox[0], 0.0)
    return float(extents[0] * extents[1] * extents[2])


def _interval_gap(a_min: float, a_max: float, b_min: float, b_max: float) -> float:
    if a_max < b_min:
        return float(b_min - a_max)
    if b_max < a_min:
        return float(a_min - b_max)
    return 0.0


def _interval_overlap(a_min: float, a_max: float, b_min: float, b_max: float) -> float:
    return max(0.0, float(min(a_max, b_max) - max(a_min, b_min)))


def _bbox_face_adjacency_score(
    bbox1: np.ndarray,
    bbox2: np.ndarray,
    gap_tolerance: float = FACE_GAP_TOLERANCE,
    min_overlap_abs: float = FACE_MIN_OVERLAP,
) -> float:
    bbox1 = np.asarray(bbox1, dtype=float)
    bbox2 = np.asarray(bbox2, dtype=float)

    best_score = -1.0
    for axis in range(3):
        gap = _interval_gap(bbox1[0][axis], bbox1[1][axis], bbox2[0][axis], bbox2[1][axis])
        if gap > gap_tolerance:
            continue

        overlap_scores = []
        for other_axis in range(3):
            if other_axis == axis:
                continue
            overlap = _interval_overlap(
                bbox1[0][other_axis], bbox1[1][other_axis],
                bbox2[0][other_axis], bbox2[1][other_axis],
            )
            overlap_scores.append(float(overlap))

        if len(overlap_scores) != 2:
            continue
        if min(overlap_scores) < min_overlap_abs:
            continue

        # Use actual shared face area instead of overlap-length sum.
        # This rejects edge/point touching and prefers large contacting faces.
        score = overlap_scores[0] * overlap_scores[1] - gap * 10.0
        best_score = max(best_score, score)

    return best_score


def _bbox_expand_ratio(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    merged = _merge_bbox_arrays([bbox1, bbox2])
    denom = max(_bbox_volume(bbox1) + _bbox_volume(bbox2), 1e-6)
    return _bbox_volume(merged) / denom


def _bbox_intersection_volume(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    bbox1 = np.asarray(bbox1, dtype=float)
    bbox2 = np.asarray(bbox2, dtype=float)
    overlap = np.maximum(np.minimum(bbox1[1], bbox2[1]) - np.maximum(bbox1[0], bbox2[0]), 0.0)
    return float(overlap[0] * overlap[1] * overlap[2])


def _bbox_has_positive_volume_overlap(bbox1: np.ndarray, bbox2: np.ndarray, tol: float = 1e-6) -> bool:
    return _bbox_intersection_volume(bbox1, bbox2) > tol


def _bbox_min_distance(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    bbox1 = np.asarray(bbox1, dtype=float)
    bbox2 = np.asarray(bbox2, dtype=float)
    gap_vec = np.maximum(np.maximum(bbox1[0] - bbox2[1], bbox2[0] - bbox1[1]), 0.0)
    return float(np.linalg.norm(gap_vec))


def _bool_op_name_from_operation(operation: int) -> str:
    operation_name = EXTRUDE_OPERATIONS[int(operation)]
    if operation_name == "CutFeatureOperation":
        return "cut"
    if operation_name == "IntersectFeatureOperation":
        return "intersect"
    return "union"


def _bool_op_name(extrude_op) -> str:
    return _bool_op_name_from_operation(extrude_op.operation)


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        return vec
    return vec / norm


def _plane_to_workplane_expr(
    sketch_plane,
    origin: Sequence[float],
    tol: float = 1e-6,
    formatter: Optional[FloatFormatter] = None,
) -> tuple[str, Optional[np.ndarray], float]:
    """
    Convert sketch plane definition to a CadQuery Workplane string.

    Returns tuple of (workplane_expression, 2x2 transform or None). The transform maps
    original (u, v) sketch coordinates to the canonical plane coordinates when a named
    plane ('XY'/'YZ'/'XZ') is used. The final value is +1 if the canonical plane matches
    the original normal direction, or -1 if it is flipped.
    """
    if formatter is None:
        formatter = _fmt

    origin_str = _fmt_tuple(origin, formatter=formatter)
    normal = _normalize(np.array(sketch_plane.normal, dtype=float))
    x_axis = _normalize(np.array(sketch_plane.x_axis, dtype=float))
    y_axis = _normalize(np.cross(normal, x_axis))

    for plane_name, axes in CANONICAL_PLANES.items():
        canon_normal = axes["normal"]
        canon_x = axes["x_axis"]
        canon_y = axes["y_axis"]

        if not (np.allclose(x_axis, canon_x, atol=tol) or np.allclose(x_axis, -canon_x, atol=tol)):
            continue

        if np.allclose(normal, canon_normal, atol=tol):
            normal_sign = 1.0
        elif np.allclose(normal, -canon_normal, atol=tol):
            normal_sign = -1.0
        else:
            continue

        transform = np.array([
            [float(np.dot(x_axis, canon_x)), float(np.dot(y_axis, canon_x))],
            [float(np.dot(x_axis, canon_y)), float(np.dot(y_axis, canon_y))]
        ])
        return f"cq.Workplane('{plane_name}',origin={origin_str})", transform, normal_sign

    plane_expr = (
        f"cq.Plane(origin={origin_str},"
        f"xDir={_fmt_tuple(sketch_plane.x_axis, formatter=formatter)},"
        f"normal={_fmt_tuple(sketch_plane.normal, formatter=formatter)})"
    )
    return f"cq.Workplane({plane_expr})", None, 1.0


def _apply_2d_transform(point: Sequence[float], transform: Optional[np.ndarray], offset: Optional[Sequence[float]]) -> tuple[float, float]:
    """Apply optional offset and transform to a 2D point."""
    x, y = map(float, point)
    if offset is not None:
        x += float(offset[0])
        y += float(offset[1])
    if transform is None:
        return x, y
    tx = transform[0, 0] * x + transform[0, 1] * y
    ty = transform[1, 0] * x + transform[1, 1] * y
    return tx, ty


def _fmt_point(pt: Sequence[float], formatter: Optional[FloatFormatter] = None) -> str:
    if formatter is None:
        formatter = _fmt
    return f"({formatter(pt[0])}, {formatter(pt[1])})"


def _loop_bbox_center_2d(
    loop,
    transform: Optional[np.ndarray] = None,
    offset: Optional[Sequence[float]] = None,
) -> Optional[tuple[float, float]]:
    coords: list[tuple[float, float]] = []
    for curve in getattr(loop, "children", []) or []:
        if isinstance(curve, Line):
            coords.append(_apply_2d_transform(curve.start_point, transform, offset))
            coords.append(_apply_2d_transform(curve.end_point, transform, offset))
        elif isinstance(curve, Arc):
            coords.append(_apply_2d_transform(curve.start_point, transform, offset))
            coords.append(_apply_2d_transform(curve.mid_point, transform, offset))
            coords.append(_apply_2d_transform(curve.end_point, transform, offset))
        elif isinstance(curve, Circle):
            cx, cy = _apply_2d_transform(curve.center, transform, offset)
            r = abs(float(curve.radius))
            coords.append((cx - r, cy - r))
            coords.append((cx + r, cy + r))

    if not coords:
        return None
    xs = [coord[0] for coord in coords]
    ys = [coord[1] for coord in coords]
    return (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0


def _loop_to_wire_commands(
    loop,
    transform: Optional[np.ndarray] = None,
    offset: Optional[Sequence[float]] = None,
    formatter: Optional[FloatFormatter] = None,
    outward_center: Optional[tuple[float, float]] = None,
    outward_clearance: float = 0.0,
) -> Optional[str]:
    """Convert a loop into CadQuery wire commands (traditional method, shorter code).

    Uses moveTo/lineTo/threePointArc/circle/close instead of Sketch API.
    Returns command_string or None if loop is empty.
    """
    if formatter is None:
        formatter = _fmt
    outward_clearance = max(float(outward_clearance), 0.0)

    def fmt_x(value: float) -> str:
        if outward_center is None:
            return formatter(value)
        return _fmt_outward_coord(value, outward_center[0], outward_clearance)

    def fmt_y(value: float) -> str:
        if outward_center is None:
            return formatter(value)
        return _fmt_outward_coord(value, outward_center[1], outward_clearance)

    def fmt_radius(value: float) -> str:
        if outward_center is None:
            return formatter(abs(float(value)))
        return _fmt_outward_radius(value, outward_clearance)

    def fmt_xy(point: Sequence[float]) -> str:
        return f"({fmt_x(point[0])}, {fmt_y(point[1])})"

    if not loop.children:
        return None

    # Single circle - use moveTo then circle for proper wire creation
    if len(loop.children) == 1 and isinstance(loop.children[0], Circle):
        circle = loop.children[0]
        cx, cy = _apply_2d_transform(circle.center, transform, offset)
        r = abs(float(circle.radius))
        return f".moveTo({formatter(cx)},{formatter(cy)}).circle({fmt_radius(r)})"

    # Check for axis-aligned rectangle - build as 4-line polygon
    rect = _detect_axis_aligned_rectangle(loop, transform, offset)
    if rect:
        cx, cy, width, height = rect
        x0, y0 = cx - width / 2, cy - height / 2
        x1, y1 = cx + width / 2, cy + height / 2
        return (f".moveTo({fmt_x(x0)},{fmt_y(y0)})"
                f".lineTo({fmt_x(x1)},{fmt_y(y0)})"
                f".lineTo({fmt_x(x1)},{fmt_y(y1)})"
                f".lineTo({fmt_x(x0)},{fmt_y(y1)})"
                f".close()")

    commands = []
    first_point: Optional[tuple[float, float]] = None
    current_point: Optional[tuple[float, float]] = None

    for curve in loop.children:
        if isinstance(curve, Line):
            sx, sy = _apply_2d_transform(curve.start_point, transform, offset)
            ex, ey = _apply_2d_transform(curve.end_point, transform, offset)

            # Skip zero-length lines
            if math.isclose(sx, ex, abs_tol=1e-12) and math.isclose(sy, ey, abs_tol=1e-12):
                continue

            if first_point is None:
                first_point = (sx, sy)
                commands.append(f".moveTo({fmt_x(sx)},{fmt_y(sy)})")
                current_point = (sx, sy)

            # Only add lineTo if we need to move
            if current_point is None or not (
                math.isclose(current_point[0], sx, abs_tol=1e-9) and
                math.isclose(current_point[1], sy, abs_tol=1e-9)
            ):
                commands.append(f".lineTo({fmt_x(sx)},{fmt_y(sy)})")

            commands.append(f".lineTo({fmt_x(ex)},{fmt_y(ey)})")
            current_point = (ex, ey)

        elif isinstance(curve, Arc):
            sx, sy = _apply_2d_transform(curve.start_point, transform, offset)
            mx, my = _apply_2d_transform(curve.mid_point, transform, offset)
            ex, ey = _apply_2d_transform(curve.end_point, transform, offset)

            # Skip zero-length arcs (start equals end)
            if math.isclose(sx, ex, abs_tol=1e-9) and math.isclose(sy, ey, abs_tol=1e-9):
                continue

            if first_point is None:
                first_point = (sx, sy)
                commands.append(f".moveTo({fmt_x(sx)},{fmt_y(sy)})")
                current_point = (sx, sy)

            # Move to start if needed
            if current_point is None or not (
                math.isclose(current_point[0], sx, abs_tol=1e-9) and
                math.isclose(current_point[1], sy, abs_tol=1e-9)
            ):
                commands.append(f".lineTo({fmt_x(sx)},{fmt_y(sy)})")

            # Check if three points are collinear (cross product near zero)
            # If collinear, use lineTo instead of threePointArc
            cross = (mx - sx) * (ey - sy) - (my - sy) * (ex - sx)
            if math.isclose(cross, 0.0, abs_tol=1e-6):
                # Points are collinear, use lineTo
                commands.append(f".lineTo({fmt_x(ex)},{fmt_y(ey)})")
            else:
                # threePointArc only needs mid and end points (start is current position)
                commands.append(
                    f".threePointArc({fmt_xy((mx, my))},"
                    f"{fmt_xy((ex, ey))})"
                )
            current_point = (ex, ey)

        elif isinstance(curve, Circle):
            cx, cy = _apply_2d_transform(curve.center, transform, offset)
            r = abs(float(curve.radius))
            if first_point is None:
                first_point = (cx, cy)
            commands.append(f".moveTo({formatter(cx)},{formatter(cy)}).circle({fmt_radius(r)})")
            current_point = None  # Circle is self-closing

    if not commands:
        return None

    # Close polygon paths (circles are self-closing and don't need this)
    if current_point is not None:
        commands.append(".close()")

    return "".join(commands)


def _extract_3d_vertices(extrude: Extrude) -> np.ndarray:
    """Extract all 3D vertices from an extrude operation."""
    profile = deepcopy(extrude.profile)
    profile.denormalize(extrude.sketch_size)

    vertices_3d = []
    origin = np.array(extrude.sketch_pos)
    x_axis = np.array(extrude.sketch_plane.x_axis)
    y_axis = np.array(extrude.sketch_plane.y_axis)
    normal = np.array(extrude.sketch_plane.normal)

    # Get extrusion extents
    extent_one = extrude.extent_one
    extent_two = extrude.extent_two
    if extrude.extent_type == EXTENT_TYPE.index("SymmetricFeatureExtentType"):
        extents = [0, extent_one, -extent_one]
    elif extrude.extent_type == EXTENT_TYPE.index("TwoSidesFeatureExtentType"):
        extents = [0, extent_one, -extent_two]
    else:
        extents = [0, extent_one]

    for loop in profile.children:
        for curve in loop.children:
            # Collect 2D points
            points_2d = []
            if isinstance(curve, Line):
                points_2d.extend([curve.start_point, curve.end_point])
            elif isinstance(curve, Arc):
                points_2d.extend([curve.start_point, curve.mid_point, curve.end_point])
            elif isinstance(curve, Circle):
                # Sample circle at 8 points
                cx, cy = curve.center
                r = abs(float(curve.radius))
                for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
                    points_2d.append((cx + r * np.cos(angle), cy + r * np.sin(angle)))

            # Convert to 3D at all extrusion levels
            for pt_2d in points_2d:
                for extent in extents:
                    pt_3d = origin + pt_2d[0] * x_axis + pt_2d[1] * y_axis + extent * normal
                    vertices_3d.append(pt_3d)

    return np.array(vertices_3d) if vertices_3d else np.empty((0, 3))

def _group_by_shared_vertices(extrudes: list) -> list[list[int]]:
    """Group sketch indices that share vertices using Union-Find.

    Returns list of groups, where each group is a list of sketch indices.
    Sketches are considered connected if they share vertices.
    """
    if len(extrudes) <= 1:
        return [[i] for i in range(len(extrudes))]

    parent = list(range(len(extrudes)))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    vertex_owner: dict[tuple[int, int, int], int] = {}
    tolerance = 1e-6
    inv_tolerance = 1.0 / tolerance

    for idx, extrude in enumerate(extrudes):
        vertices = _extract_3d_vertices(extrude)
        if len(vertices) == 0:
            continue
        vertex_keys = {
            tuple(int(round(float(coord) * inv_tolerance)) for coord in vertex)
            for vertex in vertices
        }
        for key in vertex_keys:
            owner = vertex_owner.get(key)
            if owner is None:
                vertex_owner[key] = idx
            else:
                union(owner, idx)

    groups_dict = {}
    for i in range(len(extrudes)):
        root = find(i)
        if root not in groups_dict:
            groups_dict[root] = []
        groups_dict[root].append(i)

    # Return sorted groups (by first element in each group)
    return sorted(groups_dict.values(), key=lambda g: g[0])


def _fmt(value: float) -> str:
    """Format floats preserving two decimal places."""
    return format(_quantize_decimal(value), f".{DECIMAL_PLACES}f")


def _fmt_outward_coord(value: float, center: float, clearance: float = 0.0) -> str:
    """Format a 2D cut coordinate so two-decimal rounding does not shrink the cutter."""
    numeric = 0.0 if math.isclose(value, 0.0, abs_tol=1e-12) else float(value)
    center = float(center)
    clearance = max(float(clearance), 0.0)
    if numeric > center:
        numeric += clearance
        rounding = ROUND_CEILING
    elif numeric < center:
        numeric -= clearance
        rounding = ROUND_FLOOR
    else:
        rounding = ROUND_HALF_EVEN
    decimal_value = Decimal(str(numeric))
    quantized = decimal_value.quantize(_DECIMAL_STEP, rounding=rounding)
    if quantized == 0:
        quantized = _ZERO_QUANTIZED
    return format(quantized, f".{DECIMAL_PLACES}f")


def _fmt_outward_radius(value: float, clearance: float = 0.0) -> str:
    numeric = abs(float(value)) + max(float(clearance), 0.0)
    decimal_value = Decimal(str(numeric))
    quantized = decimal_value.quantize(_DECIMAL_STEP, rounding=ROUND_CEILING)
    if quantized == 0:
        quantized = _ZERO_QUANTIZED
    return format(quantized, f".{DECIMAL_PLACES}f")


def _fmt_full_precision(value: float) -> str:
    """Format floats with enough digits to round-trip the original Python float."""
    numeric = 0.0 if math.isclose(value, 0.0, abs_tol=1e-12) else float(value)
    return format(numeric, f".{FULL_PRECISION_SIG_DIGITS}g")


def _fmt_tuple(vec: Sequence[float], formatter: FloatFormatter = _fmt) -> str:
    return f"({', '.join(formatter(num) for num in vec)})"


def _coerce_entity_ids(raw_value) -> list[str]:
    """Normalize entity references to a flat list of string IDs."""
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        return [raw_value]
    if isinstance(raw_value, dict):
        candidates = [raw_value.get("entity"), raw_value.get("id"), raw_value.get("name")]
        return [item for item in candidates if isinstance(item, str)]
    if isinstance(raw_value, Sequence):
        entity_ids: list[str] = []
        for entry in raw_value:
            entity_ids.extend(_coerce_entity_ids(entry))
        return entity_ids
    return []


def _iter_extrude_entities(sequence_data: Sequence[dict]) -> Iterable[str]:
    """Yield extrude entity IDs, supporting steps that reference multiple entities."""
    for item in sequence_data or []:
        if not isinstance(item, dict) or item.get("type") != "ExtrudeFeature":
            continue

        for entity_id in _extract_sequence_item_entity_ids(item):
            yield entity_id


def _extract_sequence_item_entity_ids(item: dict) -> list[str]:
    """Return the entity IDs referenced by an ExtrudeFeature item."""
    entities_field = item.get("entities")
    if entities_field is not None:
        entity_ids = _coerce_entity_ids(entities_field)
        if entity_ids:
            return entity_ids
    return _coerce_entity_ids(item.get("entity"))


def _collect_previous_sketch_ids(sequence_data: Sequence[dict], stop_index: int) -> list[str]:
    """Collect sketch entity IDs that appear before the given sequence index."""
    sketch_ids: list[str] = []
    seen: set[str] = set()
    for idx, item in enumerate(sequence_data or []):
        if idx >= int(stop_index):
            break
        if not isinstance(item, dict) or item.get("type") != "Sketch":
            continue
        entity_id = item.get("entity")
        if not isinstance(entity_id, str) or entity_id in seen:
            continue
        seen.add(entity_id)
        sketch_ids.append(entity_id)
    return sketch_ids


def _collect_sequence_sketch_ids(sequence_data: Sequence[dict]) -> list[str]:
    """Collect sketch entity IDs from sequence in order of appearance."""
    sketch_ids: list[str] = []
    seen: set[str] = set()
    for item in sequence_data or []:
        if not isinstance(item, dict) or item.get("type") != "Sketch":
            continue
        entity_id = item.get("entity")
        if not isinstance(entity_id, str) or entity_id in seen:
            continue
        seen.add(entity_id)
        sketch_ids.append(entity_id)
    return sketch_ids


def _collect_entity_sketch_ids(entities: dict) -> list[str]:
    """Collect sketch entity IDs from the entities table."""
    sketch_ids: list[str] = []
    for entity_id, entity in entities.items():
        if not isinstance(entity, dict):
            continue
        if entity.get("type") != "Sketch":
            continue
        sketch_ids.append(entity_id)
    return sketch_ids


def _infer_extrude_profiles_from_sequence(payload: dict) -> None:
    """Fill empty extrude profiles from preceding sketch steps when possible."""
    entities = payload.get("entities")
    sequence_data = payload.get("sequence")
    if not isinstance(entities, dict) or not isinstance(sequence_data, Sequence):
        return

    for seq_idx, item in enumerate(sequence_data):
        if not isinstance(item, dict) or item.get("type") != "ExtrudeFeature":
            continue

        extrude_ids = _extract_sequence_item_entity_ids(item)
        if not extrude_ids:
            continue

        preceding_sketch_ids = _collect_previous_sketch_ids(sequence_data, seq_idx)
        sequence_sketch_ids = _collect_sequence_sketch_ids(sequence_data)
        entity_sketch_ids = _collect_entity_sketch_ids(entities)

        candidate_sketch_ids = list(preceding_sketch_ids)
        if len(candidate_sketch_ids) == 0 and len(sequence_sketch_ids) == 1:
            candidate_sketch_ids = list(sequence_sketch_ids)
        if len(candidate_sketch_ids) == 0 and len(entity_sketch_ids) == 1:
            candidate_sketch_ids = list(entity_sketch_ids)
        if len(candidate_sketch_ids) == 0:
            continue

        for extrude_id in extrude_ids:
            extrude_entity = entities.get(extrude_id)
            if not isinstance(extrude_entity, dict):
                continue
            if extrude_entity.get("type") != "ExtrudeFeature":
                continue

            profiles = extrude_entity.get("profiles")
            if isinstance(profiles, list) and len(profiles) > 0:
                continue

            inferred_profiles = []
            for sketch_id in candidate_sketch_ids:
                sketch_entity = entities.get(sketch_id)
                if not isinstance(sketch_entity, dict):
                    continue
                if sketch_entity.get("type") != "Sketch":
                    continue
                sketch_profiles = sketch_entity.get("profiles")
                if not isinstance(sketch_profiles, dict):
                    continue
                for profile_id in sketch_profiles.keys():
                    inferred_profiles.append({
                        "sketch": sketch_id,
                        "profile": profile_id,
                    })

            if len(inferred_profiles) > 0:
                extrude_entity["profiles"] = inferred_profiles


def _extract_bbox(payload: dict) -> Optional[np.ndarray]:
    bbox_info = payload.get("properties", {}).get("bounding_box")
    if not bbox_info:
        return None
    max_point = np.array([
        bbox_info["max_point"].get("x", 0.0),
        bbox_info["max_point"].get("y", 0.0),
        bbox_info["max_point"].get("z", 0.0),
    ])
    min_point = np.array([
        bbox_info["min_point"].get("x", 0.0),
        bbox_info["min_point"].get("y", 0.0),
        bbox_info["min_point"].get("z", 0.0),
    ])
    return np.stack([max_point, min_point], axis=0)


def _build_cad_sequence_with_entity_groups(payload: dict) -> tuple[CADSequence, list[dict]]:
    payload = deepcopy(payload)
    _infer_extrude_profiles_from_sequence(payload)

    extrudes: list[Extrude] = []
    entity_groups = []
    seen_entity_ids: set[str] = set()

    for entity_id in _iter_extrude_entities(payload.get("sequence", [])):
        if entity_id in seen_entity_ids:
            continue
        seen_entity_ids.add(entity_id)

        entity_extrudes = Extrude.from_dict(payload, entity_id)
        if len(entity_extrudes) == 0:
            continue

        start_idx = len(extrudes)
        extrudes.extend(entity_extrudes)
        entity_groups.append({
            "entity_id": entity_id,
            "indices": list(range(start_idx, len(extrudes))),
            "operation": entity_extrudes[0].operation,
        })

    bbox = _extract_bbox(payload)
    return CADSequence(extrudes, bbox), entity_groups


def _get_extrude_distances(extent_type: str, extent_one: float, extent_two: float) -> list[float]:
    """Return the list of extrusion distances needed for the given extent type."""
    if extent_type == "SymmetricFeatureExtentType":
        distances = [extent_one, -extent_one]
    elif extent_type == "TwoSidesFeatureExtentType":
        distances = [extent_one, -extent_two]
    else:
        distances = [extent_one]

    # Drop near-zero distances to avoid invalid extrudes
    cleaned = []
    for dist in distances:
        if math.isclose(dist, 0.0, abs_tol=1e-9):
            continue
        cleaned.append(dist)
    return cleaned


def _loop_geometry_stats(loop) -> dict:
    stats = {
        "line_count": 0,
        "arc_count": 0,
        "circle_count": 0,
        "curve_count": len(getattr(loop, "children", []) or []),
    }
    for curve in getattr(loop, "children", []) or []:
        if isinstance(curve, Line):
            stats["line_count"] += 1
        elif isinstance(curve, Arc):
            stats["arc_count"] += 1
        elif isinstance(curve, Circle):
            stats["circle_count"] += 1
    return stats


def _append_loop_step_lines(
    lines: list[str],
    plane_expr: str,
    loop,
    step_counter: int,
    distances: Sequence[float],
    extent_type: str,
    transform: Optional[np.ndarray] = None,
    offset: Optional[Sequence[float]] = None,
    formatter: FloatFormatter = _fmt,
    step_records: Optional[list[dict]] = None,
    step_context: Optional[dict] = None,
    role: str = "body",
    solid_var: Optional[str] = None,
    solid_initialized: bool = False,
    solid_merge_op: str = "union",
    plane_distance_adjuster: Optional[Callable[[float, bool, bool], tuple[str, float]]] = None,
    cut_planar_clearance: float = 0.0,
) -> tuple[list[str], int, bool]:
    """Emit standalone step variables by drawing a loop directly from the workplane."""
    is_cut_step = solid_merge_op == "cut"
    outward_center = None
    outward_clearance = 0.0
    if is_cut_step and cut_planar_clearance > 0.0:
        outward_center = _loop_bbox_center_2d(loop, transform, offset)
        outward_clearance = cut_planar_clearance
    wire = _loop_to_wire_commands(
        loop,
        transform,
        offset,
        formatter=formatter,
        outward_center=outward_center,
        outward_clearance=outward_clearance,
    )
    if not wire or not distances:
        return [], step_counter, solid_initialized

    loop_descriptor = _describe_simple_loop(loop, transform, offset)
    loop_stats = _loop_geometry_stats(loop)
    loop_type = loop_descriptor[0] if loop_descriptor is not None else None
    loop_dims = tuple(float(value) for value in loop_descriptor[2]) if loop_descriptor is not None else None

    def append_step_merge(step_var: str) -> None:
        nonlocal solid_initialized
        if solid_var is None:
            return
        if not solid_initialized:
            lines.append(f"{solid_var} = {step_var}")
            solid_initialized = True
        elif solid_merge_op == "cut":
            lines.append(f"{solid_var} = {solid_var}.cut({step_var})")
        elif solid_merge_op == "intersect":
            lines.append(f"{solid_var} = {solid_var}.intersect({step_var})")
        else:
            lines.append(f"{solid_var} = {solid_var}.union({step_var})")

    step_vars: list[str] = []
    if extent_type == "SymmetricFeatureExtentType":
        wp_var = f"wp{step_counter}"
        step_var = f"step{step_counter}"
        distance = abs(float(distances[0]))
        step_plane_expr = plane_expr
        if plane_distance_adjuster is not None:
            step_plane_expr, distance = plane_distance_adjuster(
                distance,
                is_cut_step,
                True,
            )
        lines.append(f"{wp_var} = {step_plane_expr}")
        lines.append(
            f"{step_var} = {wp_var}{wire}.extrude({formatter(abs(distance))}, both=True)"
        )
        append_step_merge(step_var)
        step_vars.append(step_var)
        if step_records is not None:
            record = dict(step_context or {})
            record.update({
                "step_name": step_var,
                "step_index": step_counter,
                "role": role,
                "merge_op": solid_merge_op,
                "distance": abs(float(distance)),
                "both": True,
                "loop_type": loop_type,
                "loop_dims": loop_dims,
                "loop_curve_stats": loop_stats,
            })
            step_records.append(record)
        return step_vars, step_counter + 1, solid_initialized

    for distance in distances:
        wp_var = f"wp{step_counter}"
        step_var = f"step{step_counter}"
        step_plane_expr = plane_expr
        step_distance = float(distance)
        if plane_distance_adjuster is not None:
            step_plane_expr, step_distance = plane_distance_adjuster(
                step_distance,
                is_cut_step,
                False,
            )
        lines.append(f"{wp_var} = {step_plane_expr}")
        lines.append(f"{step_var} = {wp_var}{wire}.extrude({formatter(step_distance)})")
        append_step_merge(step_var)
        step_vars.append(step_var)
        if step_records is not None:
            record = dict(step_context or {})
            record.update({
                "step_name": step_var,
                "step_index": step_counter,
                "role": role,
                "merge_op": solid_merge_op,
                "distance": float(step_distance),
                "both": False,
                "loop_type": loop_type,
                "loop_dims": loop_dims,
                "loop_curve_stats": loop_stats,
            })
            step_records.append(record)
        step_counter += 1

    return step_vars, step_counter, solid_initialized


def _build_resume_solid_expr_block(
    extrude_op,
    idx: int,
    step_counter: int,
    formatter: FloatFormatter = _fmt,
    solid_var: Optional[str] = None,
    step_records: Optional[list[dict]] = None,
    step_context: Optional[dict] = None,
    solid_initialized: bool = False,
    body_merge_op: str = "union",
    hole_merge_op: str = "cut",
    cut_clearance: float = 0.0,
) -> tuple[str, Optional[str], int, bool]:
    """Build one resume-style wp/step block and merge each emitted step immediately."""
    profile = deepcopy(extrude_op.profile)
    profile.denormalize(extrude_op.sketch_size)

    if not profile.children:
        return "", None, step_counter, solid_initialized

    origin = extrude_op.sketch_pos
    plane_expr, transform, normal_sign = _plane_to_workplane_expr(
        extrude_op.sketch_plane,
        origin,
        formatter=formatter,
    )

    extent_one = extrude_op.extent_one
    extent_two = extrude_op.extent_two
    extent_type = EXTENT_TYPE[extrude_op.extent_type]
    distances = _get_extrude_distances(extent_type, extent_one, extent_two)
    if normal_sign != 1.0:
        distances = [dist * normal_sign for dist in distances]
    if not distances:
        return "", None, step_counter, solid_initialized

    cut_clearance = max(float(cut_clearance), 0.0)
    workplane_normal = _normalize(np.asarray(extrude_op.sketch_plane.normal, dtype=float)) * float(normal_sign)

    def adjust_cut_plane(distance: float, is_cut: bool, both: bool) -> tuple[str, float]:
        if not is_cut or cut_clearance <= 0.0:
            return plane_expr, distance
        if both:
            return plane_expr, abs(float(distance)) + cut_clearance

        sign = 1.0 if float(distance) >= 0.0 else -1.0
        shifted_origin = np.asarray(origin, dtype=float) - sign * workplane_normal * cut_clearance
        shifted_plane_expr, _, _ = _plane_to_workplane_expr(
            extrude_op.sketch_plane,
            shifted_origin,
            formatter=formatter,
        )
        return shifted_plane_expr, float(distance) + sign * cut_clearance * 2.0

    solid_var = solid_var or f"solid{idx}"
    base_context = dict(step_context or {})
    base_context.setdefault("sequence_index", int(idx))
    base_context.update({
        "extent_type": extent_type,
        "extent_one": float(extent_one),
        "extent_two": float(extent_two),
        "plane_normal": tuple(float(v) for v in extrude_op.sketch_plane.normal),
        "plane_x_axis": tuple(float(v) for v in extrude_op.sketch_plane.x_axis),
    })

    lines = []
    outer_loop = profile.children[0]
    hole_loops = profile.children[1:]

    body_step_vars, step_counter, solid_initialized = _append_loop_step_lines(
        lines,
        plane_expr,
        outer_loop,
        step_counter,
        distances,
        extent_type,
        transform=transform,
        formatter=formatter,
        step_records=step_records,
        step_context=base_context,
        role="body",
        solid_var=solid_var,
        solid_initialized=solid_initialized,
        solid_merge_op=body_merge_op,
        plane_distance_adjuster=adjust_cut_plane,
        cut_planar_clearance=cut_clearance,
    )
    if not body_step_vars:
        return "", None, step_counter, solid_initialized

    for hole_loop in hole_loops:
        _, step_counter, solid_initialized = _append_loop_step_lines(
            lines,
            plane_expr,
            hole_loop,
            step_counter,
            distances,
            extent_type,
            transform=transform,
            formatter=formatter,
            step_records=step_records,
            step_context=base_context,
            role="hole",
            solid_var=solid_var,
            solid_initialized=solid_initialized,
            solid_merge_op=hole_merge_op,
            plane_distance_adjuster=adjust_cut_plane,
            cut_planar_clearance=cut_clearance,
        )

    return "\n".join(lines), solid_var, step_counter, solid_initialized


def _sequence_to_code_for_indices(
    cad_seq: CADSequence,
    indices: Sequence[int],
    strict_initial_bool: bool = False,
    formatter: FloatFormatter = _fmt_full_precision,
) -> Optional[str]:
    """Build executable code for a subset of CADSequence extrudes using current step syntax."""
    ordered = [int(idx) for idx in indices]
    if not ordered:
        return "import cadquery as cq"

    if strict_initial_bool and _bool_op_name(cad_seq.seq[ordered[0]]) in {"cut", "intersect"}:
        return None

    lines = ["import cadquery as cq"]
    has_result = False
    step_counter = 0
    for idx in ordered:
        block, solid_ref, step_counter, _ = _build_resume_solid_expr_block(
            cad_seq.seq[idx],
            idx,
            step_counter,
            formatter=formatter,
            solid_var=f"solid{idx}",
        )
        if not block or not solid_ref:
            continue

        if lines[-1] != "import cadquery as cq":
            lines.append("")
        lines.append(block)

        operation = _bool_op_name(cad_seq.seq[idx])
        if not has_result:
            lines.append(f"solid = {solid_ref}")
            has_result = True
        elif operation == "cut":
            lines.append(f"solid = solid.cut({solid_ref})")
        elif operation == "intersect":
            lines.append(f"solid = solid.intersect({solid_ref})")
        else:
            lines.append(f"solid = solid.union({solid_ref})")

    return "\n".join(lines)


def _sequence_to_code_verbose_by_entity_groups(
    cad_seq: CADSequence,
    entity_groups: Sequence[dict],
    formatter: FloatFormatter = _fmt,
    step_records: Optional[list[dict]] = None,
    direct_solid: bool = True,
    cut_clearance: Optional[float] = None,
) -> str:
    lines = ["import cadquery as cq"]
    loop_counter = 0
    has_result = False
    if cut_clearance is None:
        cut_clearance = _default_cut_clearance(formatter, direct_solid)

    for entity_idx, group in enumerate(entity_groups):
        group_indices = [int(idx) for idx in group.get("indices", [])]
        extrudes = [cad_seq.seq[idx] for idx in group_indices]
        if not extrudes:
            continue

        body_merge_op = _bool_op_name_from_operation(group.get("operation"))
        hole_merge_op = "cut"
        target_solid = "solid" if direct_solid else f"solid{entity_idx}"
        target_initialized = has_result if direct_solid else False
        entity_group = group or {}

        for group_idx, sketch_indices in enumerate(_group_by_shared_vertices(list(extrudes))):
            for sketch_idx in sketch_indices:
                solid_idx = int(group_indices[sketch_idx])
                step_context = {
                    "entity_index": int(entity_idx),
                    "entity_id": entity_group.get("entity_id"),
                    "entity_operation": entity_group.get("operation"),
                    "sequence_index": solid_idx,
                    "group_index": int(group_idx),
                }
                block, solid_ref, loop_counter, target_initialized = _build_resume_solid_expr_block(
                    extrudes[sketch_idx],
                    solid_idx,
                    loop_counter,
                    formatter=formatter,
                    solid_var=target_solid,
                    step_records=step_records,
                    step_context=step_context,
                    solid_initialized=target_initialized,
                    body_merge_op=body_merge_op if direct_solid else "union",
                    hole_merge_op=hole_merge_op,
                    cut_clearance=float(cut_clearance),
                )
                if not block or not solid_ref:
                    continue
                if direct_solid:
                    has_result = target_initialized
                if lines[-1] != "":
                    lines.append("")
                lines.append(block)

        if direct_solid or not target_initialized:
            continue

        operation_name = EXTRUDE_OPERATIONS[group.get("operation")]
        if not has_result:
            lines.append(f"solid = {target_solid}")
            has_result = True
        elif operation_name in ("NewBodyFeatureOperation", "JoinFeatureOperation"):
            lines.append(f"solid = solid.union({target_solid})")
        elif operation_name == "CutFeatureOperation":
            lines.append(f"solid = solid.cut({target_solid})")
        elif operation_name == "IntersectFeatureOperation":
            lines.append(f"solid = solid.intersect({target_solid})")
        else:
            lines.append(f"solid = solid.union({target_solid})")

    return "\n".join(lines)


def json2cadquery(
    json_path: str,
    output_path: Optional[str] = None,
    normalize: bool = True,
    by_entity: bool = False,
    scale: float = 1.0,
    verbose: bool = False,
) -> str:
    """
    Convert DeepCAD JSON to CadQuery code.

    Args:
        json_path: Path to DeepCAD JSON file
        output_path: Optional output path for generated Python script
        normalize: Whether to normalize to the fixed symmetric range [-100, 100]
        by_entity: Deprecated compatibility flag; output always preserves entity boundaries.
        scale: Uniform factor applied after optional normalization (default: 1.0)
        verbose: Deprecated compatibility flag; output always uses wp/step/solid definitions.

    Returns:
        Generated CadQuery code as string
    """
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    cad_seq, entity_groups = _build_cad_sequence_with_entity_groups(payload)
    if normalize:
        _normalize_to_random_range(cad_seq, scale)
    elif not math.isclose(scale, 1.0):
        cad_seq.transform(0.0, scale)

    code = _sequence_to_code_verbose_by_entity_groups(cad_seq, entity_groups)

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code, encoding="utf-8")

    return code


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CadQuery code from DeepCAD JSON files.")
    parser.add_argument("--json", required=True, help="Path to DeepCAD JSON file")
    parser.add_argument("--output", "-o", default=None, help="Output Python file path")
    parser.add_argument("--output-dir", default=None, help="Output directory for full generation (py, stl, parts, bbox)")
    parser.add_argument("--no-normalize", action="store_true", help="Skip normalization")
    parser.add_argument("--by-entity", action="store_true", help="Compatibility flag; output is always entity grouped")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Compatibility flag; output always uses wp/step/solid definitions",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Uniform scale factor applied after optional normalization (default: 1.0)",
    )
    return parser.parse_args(argv)


def _get_bbox_array_from_shape(shape) -> np.ndarray:
    """Extract a high-precision bbox array from a CadQuery/OCC shape."""
    bb = shape.BoundingBox()
    return np.array(
        [
            [float(bb.xmin), float(bb.ymin), float(bb.zmin)],
            [float(bb.xmax), float(bb.ymax), float(bb.zmax)],
        ],
        dtype=float,
    )


def _get_bbox_array_from_solid(solid) -> np.ndarray:
    """Extract a high-precision bbox array from a CadQuery solid/workplane."""
    return _get_bbox_array_from_shape(solid.val())


def _empty_exec_artifacts() -> dict:
    return {
        "valid": False,
        "bbox": None,
        "bbox_np": None,
        "volume": 0.0,
        "solid": None,
        "step_payload": [],
        "step_artifacts": {},
        "entity_artifacts": {},
    }


def _shape_volume(shape) -> float:
    try:
        return float(shape.Volume())
    except Exception:
        return 0.0


def _solid_artifact(solid, include_volume: bool = False, keep_solid: bool = True) -> dict:
    shape = solid.val()
    bbox_np = _get_bbox_array_from_shape(shape)
    artifact = {
        "bbox_np": bbox_np,
        "bbox": _bbox_array_to_list(bbox_np),
    }
    if keep_solid:
        artifact["solid"] = solid
        artifact["shape"] = shape
    if include_volume:
        artifact["volume"] = _shape_volume(shape)
    return artifact


def _execute_code_artifacts(
    cq,
    code: Optional[str],
    step_names: Optional[Sequence[str]] = None,
    entity_names: Optional[Sequence[str]] = None,
    collect_volume: bool = True,
    keep_solid: bool = True,
    collect_step_payload: bool = True,
) -> dict:
    """Execute CadQuery code once and optionally collect per-step bbox payload."""
    if not code:
        return _empty_exec_artifacts()

    try:
        exec_globals = {"cq": cq}
        exec(code, exec_globals)
        solid_obj = exec_globals.get("solid", None)
        if solid_obj is None:
            return _empty_exec_artifacts()

        solid_artifact = _solid_artifact(
            solid_obj,
            include_volume=collect_volume,
            keep_solid=keep_solid,
        )

        step_payload = []
        step_artifacts = {}
        if step_names:
            for step_name in step_names:
                step_obj = exec_globals.get(step_name)
                if step_obj is None:
                    continue
                artifact = _solid_artifact(
                    step_obj,
                    include_volume=collect_volume,
                    keep_solid=keep_solid,
                )
                if collect_step_payload:
                    step_payload.append((step_name, artifact["bbox"]))
                step_artifacts[step_name] = artifact

        entity_artifacts = {}
        if entity_names:
            for entity_name in entity_names:
                entity_obj = exec_globals.get(entity_name)
                if entity_obj is None:
                    continue
                entity_artifacts[entity_name] = _solid_artifact(
                    entity_obj,
                    include_volume=collect_volume,
                    keep_solid=keep_solid,
                )

        return {
            "valid": True,
            "bbox": solid_artifact["bbox"],
            "bbox_np": solid_artifact["bbox_np"],
            "volume": float(solid_artifact.get("volume", 0.0)),
            "solid": solid_obj if keep_solid else None,
            "step_payload": step_payload,
            "step_artifacts": step_artifacts,
            "entity_artifacts": entity_artifacts,
        }
    except Exception:
        return _empty_exec_artifacts()


def _rounded_signature(values: Sequence[float], ndigits: int = 6) -> tuple[float, ...]:
    return tuple(round(float(value), ndigits) for value in values)


def _step_plane_extent_signature(record: dict) -> tuple:
    return (
        _rounded_signature(record.get("plane_normal", (0.0, 0.0, 0.0))),
        _rounded_signature(record.get("plane_x_axis", (0.0, 0.0, 0.0))),
        record.get("extent_type"),
        round(float(record.get("extent_one", 0.0)), 6),
        round(float(record.get("extent_two", 0.0)), 6),
    )


def _step_bool_op(record: dict) -> str:
    if record.get("role") == "hole":
        return "cut"
    operation = record.get("entity_operation")
    if operation is None:
        return "union"
    return _bool_op_name_from_operation(operation)


def _build_step_leaf_records(step_records: Sequence[dict], step_artifacts: dict) -> list[dict]:
    leaf_records = []
    for record in sorted(step_records, key=lambda item: int(item.get("step_index", 0))):
        step_name = record.get("step_name")
        if not isinstance(step_name, str) or step_name not in step_artifacts:
            continue
        artifact = step_artifacts[step_name]
        step_index = int(record.get("step_index", len(leaf_records)))
        bbox_np = np.asarray(artifact["bbox_np"], dtype=float)
        leaf_records.append({
            "index": step_index,
            "step_name": step_name,
            "solid_name": step_name,
            "sequence_indices": [int(record.get("sequence_index", step_index))],
            "entity_index": record.get("entity_index"),
            "entity_id": record.get("entity_id"),
            "role": record.get("role", "body"),
            "bbox_np": bbox_np,
            "bbox": _bbox_array_to_list(bbox_np),
            "bool_op": _step_bool_op(record),
            "solid": artifact.get("solid"),
            "shape": artifact.get("shape"),
            "volume": float(artifact.get("volume", 0.0)),
            "signature": _step_plane_extent_signature(record),
            "loop_type": record.get("loop_type"),
            "loop_dims": record.get("loop_dims"),
            "loop_curve_stats": record.get("loop_curve_stats", {}),
        })
    return leaf_records


def _build_sequence_leaf_records(
    cad_seq: CADSequence,
    step_leaf_records: Optional[Sequence[dict]] = None,
) -> list[dict]:
    """Build one grouping leaf per CADSequence extrude, reusing step bboxes when available."""
    step_bboxes_by_sequence: dict[int, list[np.ndarray]] = {}
    for leaf in step_leaf_records or []:
        bbox_np = np.asarray(leaf["bbox_np"], dtype=float)
        for seq_idx in leaf.get("sequence_indices", []):
            step_bboxes_by_sequence.setdefault(int(seq_idx), []).append(bbox_np)

    leaf_records = []
    for idx, extrude_op in enumerate(cad_seq.seq):
        step_bboxes = step_bboxes_by_sequence.get(idx)
        bbox_np = _merge_bbox_arrays(step_bboxes) if step_bboxes else _get_extrude_bbox(extrude_op)
        leaf_records.append({
            "index": idx,
            "solid_name": f"solid{idx}",
            "sequence_indices": [idx],
            "bbox_np": bbox_np,
            "bbox": _bbox_array_to_list(bbox_np),
            "bool_op": _bool_op_name(extrude_op),
        })
    return leaf_records


def _make_part_from_leaf(leaf: dict) -> dict:
    return {
        "leaf_indices": [int(leaf["index"])],
        "bbox_np": np.asarray(leaf["bbox_np"], dtype=float),
        "dependency_locked": False,
    }


def _merge_parts(parts: Sequence[dict], dependency_locked: Optional[bool] = None) -> dict:
    leaf_indices = sorted({
        int(idx)
        for part in parts
        for idx in part["leaf_indices"]
    })
    bbox_np = _merge_bbox_arrays([part["bbox_np"] for part in parts])
    if dependency_locked is None:
        dependency_locked = any(bool(part.get("dependency_locked", False)) for part in parts)
    return {
        "leaf_indices": leaf_indices,
        "bbox_np": bbox_np,
        "dependency_locked": bool(dependency_locked),
    }


def _overlap_merge_score_from_bboxes(bbox_a: np.ndarray, bbox_b: np.ndarray) -> float:
    bbox_a = np.asarray(bbox_a, dtype=float)
    bbox_b = np.asarray(bbox_b, dtype=float)
    overlap_volume = _bbox_intersection_volume(bbox_a, bbox_b)
    if overlap_volume <= 1e-6:
        return -1.0

    overlap_ratio = overlap_volume / max(min(_bbox_volume(bbox_a), _bbox_volume(bbox_b)), 1e-6)
    expand_ratio = _bbox_expand_ratio(bbox_a, bbox_b)
    return overlap_ratio - expand_ratio * 0.1


def _adjacency_merge_score_from_bboxes(bbox_a: np.ndarray, bbox_b: np.ndarray) -> float:
    bbox_a = np.asarray(bbox_a, dtype=float)
    bbox_b = np.asarray(bbox_b, dtype=float)

    if _bbox_has_positive_volume_overlap(bbox_a, bbox_b):
        return -1.0

    adjacency_score = _bbox_face_adjacency_score(
        bbox_a,
        bbox_b,
        gap_tolerance=FACE_GAP_TOLERANCE,
        min_overlap_abs=FACE_MIN_OVERLAP,
    )
    if adjacency_score < 0.0:
        return -1.0

    expand_ratio = _bbox_expand_ratio(bbox_a, bbox_b)
    return adjacency_score - expand_ratio * 0.05


def _shape_distance(shape_a, shape_b) -> Optional[float]:
    if shape_a is None or shape_b is None:
        return None
    key = (id(shape_a), id(shape_b))
    if key[0] > key[1]:
        key = (key[1], key[0])
    if key in _SHAPE_DISTANCE_CACHE:
        return _SHAPE_DISTANCE_CACHE[key]
    try:
        result = shape_a.distToShape(shape_b)
    except Exception:
        _SHAPE_DISTANCE_CACHE[key] = None
        return None
    try:
        if isinstance(result, (tuple, list)):
            distance = float(result[0])
        else:
            distance = float(result)
    except Exception:
        distance = None
    _SHAPE_DISTANCE_CACHE[key] = distance
    if len(_SHAPE_DISTANCE_CACHE) > 100_000:
        _SHAPE_DISTANCE_CACHE.clear()
    return distance


def _bbox_contact_score(bbox_a: np.ndarray, bbox_b: np.ndarray) -> float:
    overlap_volume = _bbox_intersection_volume(bbox_a, bbox_b)
    if overlap_volume > 1e-6:
        return overlap_volume
    return _bbox_face_adjacency_score(
        bbox_a,
        bbox_b,
        gap_tolerance=FACE_GAP_TOLERANCE,
        min_overlap_abs=FACE_MIN_OVERLAP,
    )


def _part_bbox_center(part: dict) -> np.ndarray:
    bbox = np.asarray(part["bbox_np"], dtype=float)
    return (bbox[0] + bbox[1]) / 2.0


def _part_primary_signature(part: dict, leaf_records_by_index: dict) -> Optional[tuple]:
    counts: dict[tuple, int] = {}
    for leaf_idx in part["leaf_indices"]:
        leaf = leaf_records_by_index[int(leaf_idx)]
        if leaf.get("bool_op") == "cut":
            continue
        signature = leaf.get("signature")
        if signature is None:
            continue
        counts[signature] = counts.get(signature, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda item: item[1])[0]


def _parts_share_primary_signature(parts: Sequence[dict], leaf_records_by_index: dict) -> bool:
    signatures = [
        _part_primary_signature(part, leaf_records_by_index)
        for part in parts
    ]
    signatures = [signature for signature in signatures if signature is not None]
    if len(signatures) < max(1, len(parts) - 1):
        return False
    return len(set(signatures)) == 1


def _part_body_leaf_records(part: dict, leaf_records_by_index: dict) -> list[dict]:
    return [
        leaf_records_by_index[int(idx)]
        for idx in part["leaf_indices"]
        if leaf_records_by_index[int(idx)].get("bool_op") != "cut"
    ]


def _part_body_bbox(part: dict, leaf_records_by_index: dict) -> np.ndarray:
    body_bboxes = [
        np.asarray(leaf["bbox_np"], dtype=float)
        for leaf in _part_body_leaf_records(part, leaf_records_by_index)
    ]
    if not body_bboxes:
        return np.asarray(part["bbox_np"], dtype=float)
    return _merge_bbox_arrays(body_bboxes)


def _part_body_center(part: dict, leaf_records_by_index: dict) -> np.ndarray:
    bbox = _part_body_bbox(part, leaf_records_by_index)
    return (bbox[0] + bbox[1]) / 2.0


def _part_has_round_body(part: dict, leaf_records_by_index: dict) -> bool:
    body_leaves = _part_body_leaf_records(part, leaf_records_by_index)
    if not body_leaves:
        return False

    body_bbox = _part_body_bbox(part, leaf_records_by_index)
    bbox_extents = np.maximum(body_bbox[1] - body_bbox[0], 0.0)
    planar_axes = np.argsort(bbox_extents)[-2:]
    planar_extents = bbox_extents[planar_axes]
    min_extent = max(float(np.min(planar_extents)), 1e-6)
    aspect = float(np.max(planar_extents)) / min_extent

    for leaf in body_leaves:
        loop_type = leaf.get("loop_type")
        if loop_type == "rect":
            continue
        if loop_type == "circle":
            return aspect <= STAR_MAX_ROUND_HUB_ASPECT

        stats = leaf.get("loop_curve_stats") or {}
        if int(stats.get("circle_count", 0)) > 0:
            return aspect <= STAR_MAX_ROUND_HUB_ASPECT
        if int(stats.get("arc_count", 0)) >= 2:
            return aspect <= STAR_MAX_ROUND_HUB_ASPECT
        if int(stats.get("curve_count", 0)) >= 8 and aspect <= 1.2:
            return True

    return False


def _part_star_adjacency_edge(part_a: dict, part_b: dict, leaf_records_by_index: dict) -> Optional[dict]:
    best_edge = None
    best_key = None

    for leaf_idx_a in part_a["leaf_indices"]:
        leaf_a = leaf_records_by_index[int(leaf_idx_a)]
        if leaf_a.get("bool_op") == "cut":
            continue
        bbox_a = np.asarray(leaf_a["bbox_np"], dtype=float)
        shape_a = leaf_a.get("shape")

        for leaf_idx_b in part_b["leaf_indices"]:
            leaf_b = leaf_records_by_index[int(leaf_idx_b)]
            if leaf_b.get("bool_op") == "cut":
                continue
            bbox_b = np.asarray(leaf_b["bbox_np"], dtype=float)
            bbox_gap = _bbox_min_distance(bbox_a, bbox_b)
            if bbox_gap > SHAPE_NEAR_TOLERANCE:
                continue

            distance = _shape_distance(shape_a, leaf_b.get("shape"))
            contact_score = _bbox_contact_score(bbox_a, bbox_b)
            if distance is not None:
                if distance > SHAPE_NEAR_TOLERANCE and contact_score < 0.0:
                    continue
                effective_distance = 0.0 if distance <= SHAPE_CONTACT_TOLERANCE else float(distance)
                fallback_rank = 0
            else:
                if contact_score < 0.0:
                    continue
                effective_distance = float(bbox_gap)
                fallback_rank = 1

            key = (
                fallback_rank,
                effective_distance,
                -float(max(contact_score, 0.0)),
                abs(int(leaf_idx_a) - int(leaf_idx_b)),
            )
            if best_key is None or key < best_key:
                best_key = key
                best_edge = {
                    "distance": effective_distance,
                    "contact_score": float(contact_score),
                    "fallback_rank": fallback_rank,
                }

    return best_edge


def _angle_coverage(angles: np.ndarray) -> float:
    if len(angles) < 2:
        return 0.0
    angles = np.sort(np.mod(angles, 2.0 * math.pi))
    gaps = np.diff(np.concatenate([angles, angles[:1] + 2.0 * math.pi]))
    return float(2.0 * math.pi - np.max(gaps))


def _is_circular_star_distribution(
    hub_part: dict,
    satellite_parts: Sequence[dict],
    leaf_records_by_index: dict,
) -> bool:
    if len(satellite_parts) < STAR_MIN_SATELLITES:
        return False
    if not _part_has_round_body(hub_part, leaf_records_by_index):
        return False
    if not _parts_share_primary_signature(satellite_parts, leaf_records_by_index):
        return False

    hub_center = _part_body_center(hub_part, leaf_records_by_index)
    centers = np.stack([_part_bbox_center(part) for part in satellite_parts], axis=0)
    deltas = centers - hub_center
    variances = np.var(deltas, axis=0)
    axes = np.argsort(variances)[-2:]
    projected = deltas[:, axes]
    radii = np.linalg.norm(projected, axis=1)
    mean_radius = float(np.mean(radii))
    if mean_radius <= 1e-6:
        return False
    radius_cv = float(np.std(radii) / mean_radius)
    if radius_cv > STAR_MAX_RADIUS_CV:
        return False

    angles = np.arctan2(projected[:, 1], projected[:, 0])
    if _angle_coverage(angles) < STAR_MIN_ANGLE_COVERAGE:
        return False

    satellite_volumes = np.array([
        max(
            sum(float(leaf_records_by_index[int(idx)].get("volume", 0.0)) for idx in part["leaf_indices"]),
            _bbox_volume(np.asarray(part["bbox_np"], dtype=float)),
        )
        for part in satellite_parts
    ], dtype=float)
    mean_volume = float(np.mean(satellite_volumes))
    if mean_volume > 1e-9 and float(np.std(satellite_volumes) / mean_volume) > STAR_MAX_SATELLITE_VOLUME_CV:
        return False

    hub_bbox = _part_body_bbox(hub_part, leaf_records_by_index)
    hub_extents = np.maximum(hub_bbox[1] - hub_bbox[0], 0.0)[axes]
    min_extent = max(float(np.min(hub_extents)), 1e-6)
    if float(np.max(hub_extents)) / min_extent > STAR_MAX_ROUND_HUB_ASPECT:
        return False

    return True


def _part_sort_key(part: dict) -> int:
    return min(int(idx) for idx in part["leaf_indices"])


def _part_contains_cut(part: dict, leaf_records_by_index: dict[int, dict]) -> bool:
    return any(
        leaf_records_by_index[int(idx)].get("bool_op") == "cut"
        for idx in part["leaf_indices"]
    )


def _part_center_distance(part_a: dict, part_b: dict) -> float:
    return float(np.linalg.norm(_part_bbox_center(part_a) - _part_bbox_center(part_b)))


def _edge_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _free_part_edge_weight(
    part_a: dict,
    part_b: dict,
    leaf_records_by_index: dict[int, dict],
) -> Optional[float]:
    """Return a merge edge weight for two free non-cut parts, or None."""
    best_key = None

    for leaf_idx_a in part_a["leaf_indices"]:
        leaf_a = leaf_records_by_index[int(leaf_idx_a)]
        if leaf_a.get("bool_op") == "cut":
            continue
        bbox_a = np.asarray(leaf_a["bbox_np"], dtype=float)

        for leaf_idx_b in part_b["leaf_indices"]:
            leaf_b = leaf_records_by_index[int(leaf_idx_b)]
            if leaf_b.get("bool_op") == "cut":
                continue
            bbox_b = np.asarray(leaf_b["bbox_np"], dtype=float)
            if _bbox_min_distance(bbox_a, bbox_b) > SHAPE_NEAR_TOLERANCE:
                continue

            overlap_score = _overlap_merge_score_from_bboxes(bbox_a, bbox_b)
            if overlap_score >= 0.0:
                candidate_key = (3, float(overlap_score))
            else:
                adjacency_score = _adjacency_merge_score_from_bboxes(bbox_a, bbox_b)
                if adjacency_score >= 0.0:
                    candidate_key = (2, float(adjacency_score))
                else:
                    distance = _shape_distance(leaf_a.get("shape"), leaf_b.get("shape"))
                    if distance is None or distance > SHAPE_CONTACT_TOLERANCE:
                        continue
                    candidate_key = (1, -float(distance))

            if best_key is None or candidate_key > best_key:
                best_key = candidate_key

    if best_key is None:
        return None
    return best_key[0] * 1_000_000.0 + best_key[1]


def _iter_bbox_candidate_part_pairs(parts: Sequence[dict]):
    indexed_bboxes = [
        (idx, np.asarray(part["bbox_np"], dtype=float))
        for idx, part in enumerate(parts)
    ]
    ordered = sorted(indexed_bboxes, key=lambda item: (item[1][0][0], item[0]))

    for pos, (idx_a, bbox_a) in enumerate(ordered):
        max_x = float(bbox_a[1][0])
        for idx_b, bbox_b in ordered[pos + 1:]:
            if float(bbox_b[0][0]) - max_x > SHAPE_NEAR_TOLERANCE:
                break
            if _bbox_min_distance(bbox_a, bbox_b) <= SHAPE_NEAR_TOLERANCE:
                yield idx_a, idx_b


def _build_free_part_graph(
    parts: Sequence[dict],
    leaf_records_by_index: dict[int, dict],
) -> tuple[dict[int, set[int]], dict[tuple[int, int], float]]:
    adjacency = {idx: set() for idx in range(len(parts))}
    edge_weights: dict[tuple[int, int], float] = {}

    for idx_a, idx_b in _iter_bbox_candidate_part_pairs(parts):
        weight = _free_part_edge_weight(parts[idx_a], parts[idx_b], leaf_records_by_index)
        if weight is None:
            continue
        adjacency[idx_a].add(idx_b)
        adjacency[idx_b].add(idx_a)
        edge_weights[_edge_key(idx_a, idx_b)] = weight

    return adjacency, edge_weights


def _connected_part_components(
    parts: Sequence[dict],
    adjacency: dict[int, set[int]],
) -> list[list[int]]:
    remaining = set(range(len(parts)))
    components: list[list[int]] = []

    while remaining:
        start = min(remaining, key=lambda idx: _part_sort_key(parts[idx]))
        stack = [start]
        remaining.remove(start)
        component = []

        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in adjacency[node]:
                if neighbor not in remaining:
                    continue
                remaining.remove(neighbor)
                stack.append(neighbor)

        component.sort(key=lambda idx: _part_sort_key(parts[idx]))
        components.append(component)

    return components


def _target_group_sizes(count: int) -> list[int]:
    if count <= 0:
        return []
    if count <= MAX_FREE_STEPS_PER_PART:
        return [count]

    quotient, remainder = divmod(count, MAX_FREE_STEPS_PER_PART)
    if MAX_FREE_STEPS_PER_PART != 3:
        sizes = [MAX_FREE_STEPS_PER_PART] * quotient
        if remainder:
            sizes.append(remainder)
        return sizes

    if remainder == 0:
        return [3] * quotient
    if remainder == 1:
        return [3] * (quotient - 1) + [2, 2]
    return [3] * quotient + [2]


def _pair_score(
    pair: tuple[int, int],
    parts: Sequence[dict],
    edge_weights: dict[tuple[int, int], float],
) -> tuple[int, float, int]:
    weight = edge_weights.get(_edge_key(pair[0], pair[1]))
    if weight is not None:
        connected = 1
        score = float(weight)
    else:
        connected = 0
        score = -_part_center_distance(parts[pair[0]], parts[pair[1]])
    span = abs(_part_sort_key(parts[pair[0]]) - _part_sort_key(parts[pair[1]]))
    return connected, score, -span


def _best_pair(
    nodes: set[int],
    parts: Sequence[dict],
    edge_weights: dict[tuple[int, int], float],
) -> list[int]:
    ordered = sorted(nodes, key=lambda idx: _part_sort_key(parts[idx]))
    if len(ordered) <= 2:
        return ordered

    best_pair = None
    best_key = None
    for pos, idx_a in enumerate(ordered):
        for idx_b in ordered[pos + 1:]:
            key = _pair_score((idx_a, idx_b), parts, edge_weights)
            if best_key is None or key > best_key:
                best_key = key
                best_pair = [idx_a, idx_b]

    return best_pair or ordered[:2]


def _best_triple(
    nodes: set[int],
    parts: Sequence[dict],
    adjacency: dict[int, set[int]],
    edge_weights: dict[tuple[int, int], float],
) -> list[int]:
    ordered = sorted(nodes, key=lambda idx: _part_sort_key(parts[idx]))
    if len(ordered) <= 3:
        return ordered

    best_triple = None
    best_key = None
    seen: set[tuple[int, int, int]] = set()
    for idx_a in ordered:
        neighbors = adjacency[idx_a] & nodes
        for idx_b in neighbors:
            candidates = (adjacency[idx_a] | adjacency[idx_b]) & nodes
            for idx_c in candidates:
                if idx_c in {idx_a, idx_b}:
                    continue
                triple = tuple(sorted((idx_a, idx_b, idx_c)))
                if triple in seen:
                    continue
                seen.add(triple)
                weights = [
                    edge_weights.get(_edge_key(triple[0], triple[1])),
                    edge_weights.get(_edge_key(triple[0], triple[2])),
                    edge_weights.get(_edge_key(triple[1], triple[2])),
                ]
                edge_count = sum(weight is not None for weight in weights)
                if edge_count < 2:
                    continue
                score = sum(float(weight) for weight in weights if weight is not None)
                span = max(_part_sort_key(parts[idx]) for idx in triple) - min(
                    _part_sort_key(parts[idx]) for idx in triple
                )
                key = (edge_count, score, -span)
                if best_key is None or key > best_key:
                    best_key = key
                    best_triple = list(triple)

    if best_triple is not None:
        return sorted(best_triple, key=lambda idx: _part_sort_key(parts[idx]))

    pair = _best_pair(nodes, parts, edge_weights)
    pair_set = set(pair)
    best_extra = None
    best_extra_key = None
    for idx in ordered:
        if idx in pair_set:
            continue
        weights = [edge_weights.get(_edge_key(idx, pair_idx)) for pair_idx in pair]
        edge_count = sum(weight is not None for weight in weights)
        score = sum(float(weight) for weight in weights if weight is not None)
        key = (edge_count, score, -min(abs(_part_sort_key(parts[idx]) - _part_sort_key(parts[pair_idx])) for pair_idx in pair))
        if best_extra_key is None or key > best_extra_key:
            best_extra_key = key
            best_extra = idx

    if best_extra is not None:
        return sorted([*pair, best_extra], key=lambda idx: _part_sort_key(parts[idx]))
    return ordered[:3]


def _best_four_as_two_pairs(
    nodes: set[int],
    parts: Sequence[dict],
    edge_weights: dict[tuple[int, int], float],
) -> list[list[int]]:
    ordered = sorted(nodes, key=lambda idx: _part_sort_key(parts[idx]))
    if len(ordered) != 4:
        return []

    a, b, c, d = ordered
    partitions = [
        [(a, b), (c, d)],
        [(a, c), (b, d)],
        [(a, d), (b, c)],
    ]
    best_partition = partitions[0]
    best_key = None
    for partition in partitions:
        pair_keys = [_pair_score(pair, parts, edge_weights) for pair in partition]
        connected_count = sum(key[0] for key in pair_keys)
        score = sum(key[1] for key in pair_keys)
        span_penalty = sum(-key[2] for key in pair_keys)
        key = (connected_count, score, -span_penalty)
        if best_key is None or key > best_key:
            best_key = key
            best_partition = partition

    return [
        sorted(list(pair), key=lambda idx: _part_sort_key(parts[idx]))
        for pair in best_partition
    ]


def _choose_group_for_size(
    nodes: set[int],
    target_size: int,
    parts: Sequence[dict],
    adjacency: dict[int, set[int]],
    edge_weights: dict[tuple[int, int], float],
) -> list[int]:
    target_size = min(int(target_size), len(nodes))
    if target_size <= 1:
        return [min(nodes, key=lambda idx: _part_sort_key(parts[idx]))]
    if target_size == 2:
        return _best_pair(nodes, parts, edge_weights)
    return _best_triple(nodes, parts, adjacency, edge_weights)


def _pack_component_parts(
    component: Sequence[int],
    parts: Sequence[dict],
    adjacency: dict[int, set[int]],
    edge_weights: dict[tuple[int, int], float],
) -> list[dict]:
    remaining = set(component)
    group_sizes = _target_group_sizes(len(remaining))
    packed_parts = []

    while remaining and group_sizes:
        if len(remaining) == 4 and len(group_sizes) >= 2 and group_sizes[0] == 2 and group_sizes[1] == 2:
            for group in _best_four_as_two_pairs(remaining, parts, edge_weights):
                packed_parts.append(_merge_parts([parts[idx] for idx in group]))
            remaining.clear()
            break

        group_size = group_sizes.pop(0)
        group = _choose_group_for_size(remaining, group_size, parts, adjacency, edge_weights)
        if not group:
            break
        packed_parts.append(_merge_parts([parts[idx] for idx in group]))
        remaining.difference_update(group)

    for idx in sorted(remaining, key=lambda item: _part_sort_key(parts[item])):
        packed_parts.append(parts[idx])

    return packed_parts


def _pack_free_parts_by_graph(
    free_parts: Sequence[dict],
    leaf_records_by_index: dict[int, dict],
) -> list[dict]:
    if len(free_parts) <= 1:
        return list(free_parts)

    adjacency, edge_weights = _build_free_part_graph(free_parts, leaf_records_by_index)
    components = _connected_part_components(free_parts, adjacency)
    packed_parts = []

    for component in components:
        if len(component) == 1:
            packed_parts.append(free_parts[component[0]])
        else:
            packed_parts.extend(_pack_component_parts(component, free_parts, adjacency, edge_weights))

    packed_parts.sort(key=_part_sort_key)
    return packed_parts


def _leaf_bbox_center(leaf: dict) -> np.ndarray:
    bbox = np.asarray(leaf["bbox_np"], dtype=float)
    return (bbox[0] + bbox[1]) / 2.0


def _radial_normal_axis(axes: Sequence[int]) -> int:
    return next(axis for axis in range(3) if axis not in set(int(item) for item in axes))


def _radial_axis_overlap_ok(leaf: dict, bbox: np.ndarray, cluster: dict) -> bool:
    bool_op = leaf.get("bool_op", "union")
    normal_axis = int(cluster["normal_axis"])
    cluster_min = float(cluster["normal_min"])
    cluster_max = float(cluster["normal_max"])
    cluster_extent = max(cluster_max - cluster_min, 1e-6)
    bbox = np.asarray(bbox, dtype=float)
    leaf_min = float(bbox[0][normal_axis])
    leaf_max = float(bbox[1][normal_axis])
    leaf_extent = max(leaf_max - leaf_min, 1e-6)
    overlap = _interval_overlap(cluster_min, cluster_max, leaf_min, leaf_max)

    # Long cutters often start from one side of a gear and extend far past it.
    # Assign them to the radial layer whose axial boundary they originate from.
    if bool_op == "cut" and leaf_extent > cluster_extent * 2.0:
        boundary_tol = max(FACE_GAP_TOLERANCE, cluster_extent * 0.05)
        return (
            overlap > FACE_MIN_OVERLAP and
            (
                abs(leaf_max - cluster_max) <= boundary_tol or
                abs(leaf_min - cluster_min) <= boundary_tol
            )
        )

    required_overlap = max(
        FACE_MIN_OVERLAP,
        min(cluster_extent, leaf_extent) * RADIAL_AXIAL_OVERLAP_RATIO,
    )
    if overlap >= required_overlap:
        return True

    # Some gears are modeled as a radial tooth layer touching a coaxial circular
    # hub layer. Keep comparable-radius circular hubs with their tooth ring, but
    # avoid swallowing a second stacked gear whose hub is much smaller/larger.
    if leaf.get("loop_type") == "circle" and bool_op != "cut":
        if _interval_gap(cluster_min, cluster_max, leaf_min, leaf_max) <= FACE_GAP_TOLERANCE:
            axes = list(cluster["axes"])
            planar_extents = np.maximum(bbox[1][axes] - bbox[0][axes], 0.0)
            planar_radius = float(np.max(planar_extents) / 2.0)
            radius_ratio = planar_radius / max(float(cluster["radius"]), 1e-6)
            return 0.45 <= radius_ratio <= 1.35

    return False


def _radial_distribution_info(leaves: Sequence[dict]) -> Optional[dict]:
    if len(leaves) < STAR_MIN_SATELLITES:
        return None

    centers = np.stack([_leaf_bbox_center(leaf) for leaf in leaves], axis=0)
    center = centers.mean(axis=0)
    deltas = centers - center
    axes = np.argsort(np.var(deltas, axis=0))[-2:]
    projected = deltas[:, axes]
    radii = np.linalg.norm(projected, axis=1)
    mean_radius = float(np.mean(radii))
    if mean_radius <= 1e-6:
        return None
    if float(np.std(radii) / mean_radius) > STAR_MAX_RADIUS_CV:
        return None

    angles = np.arctan2(projected[:, 1], projected[:, 0])
    if _angle_coverage(angles) < STAR_MIN_ANGLE_COVERAGE:
        return None

    volumes = np.array([
        max(float(leaf.get("volume", 0.0)), _bbox_volume(np.asarray(leaf["bbox_np"], dtype=float)))
        for leaf in leaves
    ], dtype=float)
    mean_volume = float(np.mean(volumes))
    if mean_volume > 1e-9 and float(np.std(volumes) / mean_volume) > STAR_MAX_SATELLITE_VOLUME_CV:
        return None

    bbox_np = _merge_bbox_arrays([np.asarray(leaf["bbox_np"], dtype=float) for leaf in leaves])
    normal_axis = _radial_normal_axis(axes)
    return {
        "leaves": list(leaves),
        "leaf_indices": [int(leaf["index"]) for leaf in leaves],
        "center": center,
        "axes": tuple(int(axis) for axis in axes),
        "normal_axis": normal_axis,
        "normal_min": float(bbox_np[0][normal_axis]),
        "normal_max": float(bbox_np[1][normal_axis]),
        "radius": mean_radius,
        "bbox_np": bbox_np,
    }


def _global_radial_infos(step_leaf_records: Sequence[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for leaf in step_leaf_records:
        if leaf.get("bool_op") == "cut":
            continue
        if leaf.get("loop_type") == "circle":
            continue
        signature = leaf.get("signature")
        if signature is None:
            continue
        groups.setdefault(signature, []).append(leaf)

    infos = []
    for leaves in groups.values():
        info = _radial_distribution_info(leaves)
        if info is not None:
            infos.append(info)
    infos.sort(key=lambda info: min(info["leaf_indices"]))
    return infos


def _radial_infos_same_frame(info: dict, cluster: dict) -> bool:
    if set(info["axes"]) != set(cluster["axes"]):
        return False
    info_min = float(info["normal_min"])
    info_max = float(info["normal_max"])
    cluster_min = float(cluster["normal_min"])
    cluster_max = float(cluster["normal_max"])
    min_extent = max(
        min(info_max - info_min, cluster_max - cluster_min),
        1e-6,
    )
    required_overlap = max(FACE_MIN_OVERLAP, min_extent * RADIAL_AXIAL_OVERLAP_RATIO)
    if _interval_overlap(info_min, info_max, cluster_min, cluster_max) < required_overlap:
        return False

    axes = list(cluster["axes"])
    distance = float(np.linalg.norm(info["center"][axes] - cluster["center"][axes]))
    tolerance = max(FACE_GAP_TOLERANCE, min(float(info["radius"]), float(cluster["radius"])) * 0.15)
    return distance <= tolerance


def _cluster_global_radial_infos(infos: Sequence[dict]) -> list[dict]:
    clusters: list[dict] = []
    for info in infos:
        matching_cluster = None
        for cluster in clusters:
            if _radial_infos_same_frame(info, cluster):
                matching_cluster = cluster
                break

        if matching_cluster is None:
            clusters.append({
                "infos": [info],
                "axes": info["axes"],
                "normal_axis": info["normal_axis"],
                "normal_min": float(info["normal_min"]),
                "normal_max": float(info["normal_max"]),
                "center": np.asarray(info["center"], dtype=float),
                "radius": float(info["radius"]),
            })
            continue

        matching_cluster["infos"].append(info)
        count = len(matching_cluster["infos"])
        matching_cluster["center"] = (
            matching_cluster["center"] * (count - 1) + np.asarray(info["center"], dtype=float)
        ) / count
        matching_cluster["radius"] = max(float(matching_cluster["radius"]), float(info["radius"]))
        matching_cluster["normal_min"] = min(float(matching_cluster["normal_min"]), float(info["normal_min"]))
        matching_cluster["normal_max"] = max(float(matching_cluster["normal_max"]), float(info["normal_max"]))

    return clusters


def _expand_radial_cluster_leaf_indices(
    cluster: dict,
    step_leaf_records: Sequence[dict],
) -> list[int]:
    leaf_indices = {
        int(idx)
        for info in cluster["infos"]
        for idx in info["leaf_indices"]
    }
    axes = list(cluster["axes"])
    center = np.asarray(cluster["center"], dtype=float)
    radius = float(cluster["radius"])
    cluster_bbox = _merge_bbox_arrays([
        np.asarray(leaf["bbox_np"], dtype=float)
        for info in cluster["infos"]
        for leaf in info["leaves"]
    ])
    center_tolerance = max(FACE_GAP_TOLERANCE, radius * 0.35)

    for leaf in step_leaf_records:
        leaf_idx = int(leaf["index"])
        if leaf_idx in leaf_indices:
            continue

        leaf_center = _leaf_bbox_center(leaf)
        planar_distance = float(np.linalg.norm(leaf_center[axes] - center[axes]))
        if planar_distance > center_tolerance:
            continue

        bbox = np.asarray(leaf["bbox_np"], dtype=float)
        if not _radial_axis_overlap_ok(leaf, bbox, cluster):
            continue

        if leaf.get("loop_type") == "circle" or leaf.get("bool_op") == "cut":
            leaf_indices.add(leaf_idx)
        elif _leaf_bboxes_connected(bbox, cluster_bbox):
            leaf_indices.add(leaf_idx)

    return sorted(leaf_indices)


def _build_radial_step_parts(step_leaf_records: Sequence[dict]) -> list[dict]:
    leaf_records_by_index = {int(leaf["index"]): leaf for leaf in step_leaf_records}
    infos = _global_radial_infos(step_leaf_records)
    if not infos:
        return []

    radial_parts = []
    for cluster in _cluster_global_radial_infos(infos):
        leaf_indices = _expand_radial_cluster_leaf_indices(cluster, step_leaf_records)
        if len(leaf_indices) < STAR_MIN_SATELLITES:
            continue
        radial_parts.append(
            _merge_parts(
                [_make_part_from_leaf(leaf_records_by_index[idx]) for idx in leaf_indices],
                dependency_locked=True,
            )
        )

    radial_parts.sort(key=_part_sort_key)
    return radial_parts


def _overlay_radial_step_parts(
    base_parts: Sequence[dict],
    radial_parts: Sequence[dict],
    leaf_records_by_index: dict[int, dict],
) -> list[dict]:
    claimed_indices = {
        int(idx)
        for radial_part in radial_parts
        for idx in radial_part["leaf_indices"]
    }

    residual_parts = []
    for part in base_parts:
        remaining_indices = [
            int(idx)
            for idx in part["leaf_indices"]
            if int(idx) not in claimed_indices
        ]
        if not remaining_indices:
            continue
        residual_parts.append(
            _merge_parts(
                [_make_part_from_leaf(leaf_records_by_index[idx]) for idx in remaining_indices],
                dependency_locked=bool(part.get("dependency_locked", False)),
            )
        )

    parts = list(radial_parts) + residual_parts
    parts.sort(key=_part_sort_key)
    return parts


def _map_sequence_parts_to_step_parts(
    sequence_parts: Sequence[dict],
    step_leaf_records: Sequence[dict],
) -> list[dict]:
    step_records_by_index = {int(leaf["index"]): leaf for leaf in step_leaf_records}
    sequence_to_step_indices: dict[int, list[int]] = {}
    for leaf in step_leaf_records:
        for seq_idx in leaf.get("sequence_indices", []):
            sequence_to_step_indices.setdefault(int(seq_idx), []).append(int(leaf["index"]))

    step_parts = []
    for part in sequence_parts:
        step_indices = sorted({
            step_idx
            for seq_idx in part["leaf_indices"]
            for step_idx in sequence_to_step_indices.get(int(seq_idx), [])
        })
        if not step_indices:
            continue
        step_parts.append({
            "leaf_indices": step_indices,
            "bbox_np": _merge_bbox_arrays([
                step_records_by_index[step_idx]["bbox_np"]
                for step_idx in step_indices
            ]),
            "dependency_locked": bool(part.get("dependency_locked", False)),
        })
    return step_parts


def _leaf_bboxes_connected(
    bbox_a: np.ndarray,
    bbox_b: np.ndarray,
    gap_tolerance: float = FACE_GAP_TOLERANCE,
) -> bool:
    bbox_a = np.asarray(bbox_a, dtype=float)
    bbox_b = np.asarray(bbox_b, dtype=float)

    if _bbox_has_positive_volume_overlap(bbox_a, bbox_b):
        return True

    adjacency_score = _bbox_face_adjacency_score(
        bbox_a,
        bbox_b,
        gap_tolerance=gap_tolerance,
        min_overlap_abs=FACE_MIN_OVERLAP,
    )
    return adjacency_score >= 0.0


def _leaf_records_connected(leaf_a: dict, leaf_b: dict) -> bool:
    bbox_a = np.asarray(leaf_a["bbox_np"], dtype=float)
    bbox_b = np.asarray(leaf_b["bbox_np"], dtype=float)
    if _leaf_bboxes_connected(bbox_a, bbox_b):
        return True

    if _bbox_min_distance(bbox_a, bbox_b) > SHAPE_NEAR_TOLERANCE:
        return False

    distance = _shape_distance(leaf_a.get("shape"), leaf_b.get("shape"))
    return distance is not None and distance <= SHAPE_NEAR_TOLERANCE


def _part_is_circular_star(part: dict, leaf_records_by_index: dict) -> bool:
    leaf_indices = sorted(int(idx) for idx in part["leaf_indices"])
    if len(leaf_indices) < STAR_MIN_SATELLITES + 1:
        return False

    leaf_parts = []
    for leaf_idx in leaf_indices:
        leaf = leaf_records_by_index[leaf_idx]
        if leaf.get("bool_op") == "cut":
            continue
        leaf_parts.append(_make_part_from_leaf(leaf))

    if len(leaf_parts) < STAR_MIN_SATELLITES + 1:
        return False

    for hub_part in leaf_parts:
        if not _part_has_round_body(hub_part, leaf_records_by_index):
            continue
        satellites = [
            candidate
            for candidate in leaf_parts
            if candidate is not hub_part
            and _part_star_adjacency_edge(hub_part, candidate, leaf_records_by_index) is not None
        ]
        if _is_circular_star_distribution(hub_part, satellites, leaf_records_by_index):
            return True

    return False


def _split_part_by_leaf_connectivity(part: dict, leaf_records_by_index: dict) -> list[dict]:
    leaf_indices = sorted(int(idx) for idx in part["leaf_indices"])
    if len(leaf_indices) <= 1:
        return [part]
    if _part_is_circular_star(part, leaf_records_by_index):
        return [part]

    parent = {idx: idx for idx in leaf_indices}

    def find(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(idx_a: int, idx_b: int) -> None:
        root_a = find(idx_a)
        root_b = find(idx_b)
        if root_a != root_b:
            parent[root_b] = root_a

    for i, idx_a in enumerate(leaf_indices):
        leaf_a = leaf_records_by_index[idx_a]
        for idx_b in leaf_indices[i + 1:]:
            leaf_b = leaf_records_by_index[idx_b]
            if _leaf_records_connected(leaf_a, leaf_b):
                union(idx_a, idx_b)

    components: dict[int, list[int]] = {}
    for idx in leaf_indices:
        root = find(idx)
        components.setdefault(root, []).append(idx)

    if len(components) == 1:
        return [part]

    split_parts = []
    dependency_locked = bool(part.get("dependency_locked", False))
    for component_leaf_indices in sorted(components.values(), key=lambda group: min(group)):
        component_bboxes = [
            np.asarray(leaf_records_by_index[idx]["bbox_np"], dtype=float)
            for idx in component_leaf_indices
        ]
        split_parts.append({
            "leaf_indices": sorted(component_leaf_indices),
            "bbox_np": _merge_bbox_arrays(component_bboxes),
            "dependency_locked": dependency_locked,
        })
    return split_parts


def _split_disconnected_parts(parts: Sequence[dict], leaf_records_by_index: dict) -> list[dict]:
    split_parts = []
    for part in parts:
        if bool(part.get("dependency_locked", False)):
            split_parts.append(part)
            continue
        split_parts.extend(_split_part_by_leaf_connectivity(part, leaf_records_by_index))
    return split_parts


def _merge_or_create_dependency_part(parts: Sequence[dict], dependency_indices: Sequence[int], leaf_records_by_index: dict) -> list[dict]:
    dep_set = {int(idx) for idx in dependency_indices}
    matching_parts = [part for part in parts if dep_set & set(part["leaf_indices"])]
    if len(matching_parts) == 0:
        matching_parts = []
        for idx in sorted(dep_set):
            leaf = leaf_records_by_index[idx]
            if leaf["bool_op"] == "cut":
                matching_parts.append({
                    "leaf_indices": [idx],
                    "bbox_np": np.asarray(leaf["bbox_np"], dtype=float),
                    "dependency_locked": True,
                })
            else:
                matching_parts.append(_make_part_from_leaf(leaf))

    merged_bbox_inputs = [part["bbox_np"] for part in matching_parts]
    for idx in sorted(dep_set):
        if not any(idx in part["leaf_indices"] for part in matching_parts):
            merged_bbox_inputs.append(leaf_records_by_index[idx]["bbox_np"])

    merged_leaf_indices = sorted(set(idx for part in matching_parts for idx in part["leaf_indices"]) | dep_set)
    merged_part = {
        "leaf_indices": merged_leaf_indices,
        "bbox_np": _merge_bbox_arrays(merged_bbox_inputs),
        "dependency_locked": True,
    }

    matching_part_ids = {id(part) for part in matching_parts}
    next_parts = [part for part in parts if id(part) not in matching_part_ids]
    next_parts.append(merged_part)
    return next_parts


def _sequence_exec_metrics_cached(
    cq,
    cad_seq: CADSequence,
    indices: Sequence[int],
    cache: dict,
) -> dict:
    key = tuple(int(idx) for idx in indices)
    if key not in cache:
        code = _sequence_to_code_for_indices(
            cad_seq,
            key,
            strict_initial_bool=True,
            formatter=_fmt_full_precision,
        )
        cache[key] = _execute_code_artifacts(cq, code, keep_solid=False)
    return cache[key]


def _sequence_dependency_window_valid(
    cq,
    cad_seq: CADSequence,
    indices: Sequence[int],
    step_index: int,
    cache: dict,
    leaf_records_by_index: dict[int, dict],
) -> bool:
    ordered = [int(idx) for idx in indices]
    if step_index not in ordered:
        return False
    if _bool_op_name(cad_seq.seq[int(step_index)]) != "cut":
        return False

    base_indices = [idx for idx in ordered if idx != int(step_index)]
    if not base_indices:
        return False

    cutter_bbox = np.asarray(leaf_records_by_index[int(step_index)]["bbox_np"], dtype=float)
    base_bboxes = [
        np.asarray(leaf_records_by_index[int(idx)]["bbox_np"], dtype=float)
        for idx in base_indices
        if int(idx) in leaf_records_by_index
    ]
    if base_bboxes and not _bbox_has_positive_volume_overlap(cutter_bbox, _merge_bbox_arrays(base_bboxes)):
        return False

    metrics = _sequence_exec_metrics_cached(cq, cad_seq, ordered, cache)
    if not metrics["valid"]:
        return False

    base_metrics = _sequence_exec_metrics_cached(cq, cad_seq, base_indices, cache)
    if not base_metrics["valid"] or base_metrics["bbox"] is None:
        return False

    base_bbox = _bbox_list_to_array(base_metrics["bbox"])
    if not _bbox_has_positive_volume_overlap(cutter_bbox, base_bbox):
        return False

    base_volume = float(base_metrics.get("volume", 0.0))
    cut_volume = float(metrics.get("volume", 0.0))
    return base_volume - cut_volume > max(1e-6, abs(base_volume) * 1e-6)


def _find_sequence_dependency_indices(
    cq,
    cad_seq: CADSequence,
    step_index: int,
    cache: dict,
    leaf_records_by_index: dict[int, dict],
) -> list[int]:
    """Find every earlier base that this cut actually affects, then fall back to a minimal window."""
    step_index = int(step_index)
    if _bool_op_name(cad_seq.seq[step_index]) != "cut":
        return [step_index]

    direct_dependency_indices = {step_index}
    for candidate_idx in range(step_index):
        candidate_idx = int(candidate_idx)
        if _bool_op_name(cad_seq.seq[candidate_idx]) == "cut":
            continue
        if _sequence_dependency_window_valid(
            cq,
            cad_seq,
            [candidate_idx, step_index],
            step_index,
            cache,
            leaf_records_by_index,
        ):
            direct_dependency_indices.add(candidate_idx)

    if len(direct_dependency_indices) > 1:
        return sorted(direct_dependency_indices)

    if _sequence_dependency_window_valid(
        cq,
        cad_seq,
        [step_index],
        step_index,
        cache,
        leaf_records_by_index,
    ):
        return [step_index]

    window = None
    for start_idx in range(step_index - 1, -1, -1):
        trial = list(range(start_idx, step_index + 1))
        if _sequence_dependency_window_valid(
            cq,
            cad_seq,
            trial,
            step_index,
            cache,
            leaf_records_by_index,
        ):
            window = trial
            break

    if window is None:
        return list(range(0, step_index + 1))

    changed = True
    while changed:
        changed = False
        for candidate in list(window[:-1]):
            trial = [idx for idx in window if idx != candidate]
            if trial and _sequence_dependency_window_valid(
                cq,
                cad_seq,
                trial,
                step_index,
                cache,
                leaf_records_by_index,
            ):
                window = trial
                changed = True
    return sorted(window)


def _build_sequence_part_groups(cq, cad_seq: CADSequence, leaf_records: Sequence[dict]) -> list[dict]:
    """Part grouping copied from the root script: cut locks, bbox overlap, bbox adjacency."""
    leaf_records_by_index = {int(leaf["index"]): leaf for leaf in leaf_records}
    parts = [_make_part_from_leaf(leaf) for leaf in leaf_records]
    exec_cache: dict = {}

    for leaf in leaf_records:
        if leaf["bool_op"] != "cut":
            continue
        dependency_indices = _find_sequence_dependency_indices(
            cq,
            cad_seq,
            int(leaf["index"]),
            exec_cache,
            leaf_records_by_index,
        )
        if dependency_indices != [int(leaf["index"])]:
            parts = _merge_or_create_dependency_part(parts, dependency_indices, leaf_records_by_index)

    locked_parts = [part for part in parts if bool(part.get("dependency_locked", False))]
    free_parts = [part for part in parts if not bool(part.get("dependency_locked", False))]
    free_parts = _pack_free_parts_by_graph(free_parts, leaf_records_by_index)

    parts = locked_parts + free_parts
    parts = _split_disconnected_parts(parts, leaf_records_by_index)
    parts.sort(key=lambda part: min(part["leaf_indices"]))
    return parts


def _step_dependency_indices(step_index: int, leaf_records_by_index: dict) -> list[int]:
    leaf = leaf_records_by_index[int(step_index)]
    if leaf["bool_op"] != "cut":
        return [int(step_index)]

    cut_bbox = np.asarray(leaf["bbox_np"], dtype=float)
    cut_shape = leaf.get("shape")
    dependency_indices = {int(step_index)}

    for candidate_idx in range(int(step_index)):
        candidate = leaf_records_by_index.get(candidate_idx)
        if candidate is None or candidate.get("bool_op") == "cut":
            continue
        candidate_bbox = np.asarray(candidate["bbox_np"], dtype=float)
        if not _bbox_has_positive_volume_overlap(cut_bbox, candidate_bbox):
            continue

        distance = _shape_distance(cut_shape, candidate.get("shape"))
        if distance is None or distance <= SHAPE_NEAR_TOLERANCE:
            dependency_indices.add(candidate_idx)

    return sorted(dependency_indices) if len(dependency_indices) > 1 else [int(step_index)]


def _build_step_part_groups(cq, cad_seq: CADSequence, step_leaf_records: Sequence[dict]) -> list[dict]:
    """Group current raw steps using sequence-level geometry, then step-level radial repair."""
    step_records_by_index = {int(leaf["index"]): leaf for leaf in step_leaf_records}
    radial_parts = _build_radial_step_parts(step_leaf_records)
    if radial_parts:
        radial_covered_indices = {
            int(idx)
            for part in radial_parts
            for idx in part["leaf_indices"]
        }
        if radial_covered_indices == set(step_records_by_index.keys()):
            radial_parts.sort(key=_part_sort_key)
            return radial_parts

    sequence_leaf_records = _build_sequence_leaf_records(cad_seq, step_leaf_records)
    sequence_parts = _build_sequence_part_groups(cq, cad_seq, sequence_leaf_records)
    parts = _map_sequence_parts_to_step_parts(sequence_parts, step_leaf_records)
    if not parts:
        parts = [_make_part_from_leaf(leaf) for leaf in step_leaf_records]
        for leaf in step_leaf_records:
            if leaf["bool_op"] != "cut":
                continue
            dependency_indices = _step_dependency_indices(int(leaf["index"]), step_records_by_index)
            if dependency_indices != [int(leaf["index"])]:
                parts = _merge_or_create_dependency_part(parts, dependency_indices, step_records_by_index)
        locked_parts = []
        free_parts = []
        for part in parts:
            if bool(part.get("dependency_locked", False)) or _part_contains_cut(part, step_records_by_index):
                locked_parts.append(_merge_parts([part], dependency_locked=True))
            else:
                free_parts.append(part)
        free_parts = _pack_free_parts_by_graph(free_parts, step_records_by_index)
        parts = locked_parts + free_parts

    if radial_parts:
        parts = _overlay_radial_step_parts(parts, radial_parts, step_records_by_index)

    parts.sort(key=_part_sort_key)
    return parts


def _serialize_root_bbox_json(
    model_bbox: Sequence[float],
    part_records: Sequence[tuple[str, Sequence[float]]],
) -> dict:
    payload = {
        "label": "SPLIT",
        "model_bbox": [round(float(v), 2) for v in model_bbox],
        "parts": {},
    }
    for part_name, part_bbox in part_records:
        payload["parts"][part_name] = [round(float(v), 2) for v in part_bbox]
    return payload


def _serialize_part_bbox_json(
    model_bbox: Sequence[float],
    step_records: Sequence[tuple[str, Sequence[float]]],
) -> dict:
    payload = {
        "label": "STOP",
        "model_bbox": [round(float(v), 2) for v in model_bbox],
        "steps": {},
    }
    for step_name, step_bbox in step_records:
        payload["steps"][step_name] = [round(float(v), 2) for v in step_bbox]
    return payload


def _prepare_point_cloud_with_normals(
    points: np.ndarray,
    normals: Optional[np.ndarray] = None,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Point cloud must have shape [N, 3], got {points.shape}")

    if normals is None:
        normals = np.zeros_like(points, dtype=np.float32)
    else:
        normals = np.asarray(normals, dtype=np.float32)
        if normals.shape != points.shape:
            raise ValueError(
                f"Point normals must have shape {points.shape}, got {normals.shape}"
            )

    point_cloud = np.concatenate([points, normals], axis=1).astype(np.float32, copy=False)
    finite_mask = np.isfinite(point_cloud).all(axis=1)
    point_cloud = point_cloud[finite_mask]
    if len(point_cloud) == 0:
        raise ValueError("No finite points available for point cloud export")
    return point_cloud


def _write_point_cloud_assets(
    points: np.ndarray,
    ply_path: Path,
    normals: Optional[np.ndarray] = None,
    npy_path: Optional[Path] = None,
) -> Path:
    point_cloud = _prepare_point_cloud_with_normals(points, normals)
    if npy_path is None:
        npy_path = ply_path.with_suffix(".npy")

    ply_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ply_path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(point_cloud)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write("end_header\n")
        np.savetxt(f, point_cloud, fmt="%.8f %.8f %.8f %.8f %.8f %.8f")

    np.save(npy_path, point_cloud.astype(np.float32, copy=False))
    return npy_path


def _load_trimesh_mesh(stl_path: Path):
    import trimesh

    mesh = trimesh.load(str(stl_path), force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return mesh


def _export_cadquery_stl(cq, solid, stl_path: Path) -> None:
    cq.exporters.export(solid, str(stl_path))


def _select_point_cloud_from_samples(
    sampled_points: np.ndarray,
    sampled_normals: Optional[np.ndarray],
    n_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    sampled_points = np.asarray(sampled_points, dtype=np.float32)
    if sampled_normals is None:
        sampled_normals = np.zeros_like(sampled_points, dtype=np.float32)
    else:
        sampled_normals = np.asarray(sampled_normals, dtype=np.float32)
        if sampled_normals.shape != sampled_points.shape:
            sampled_normals = np.zeros_like(sampled_points, dtype=np.float32)

    if sampled_points.ndim != 2 or sampled_points.shape[1] != 3 or len(sampled_points) == 0:
        raise ValueError(f"Surface samples must have shape [N, 3] with N>0, got {sampled_points.shape}")

    candidate_indices = np.arange(len(sampled_points), dtype=np.int64)
    selected_indices = _sample_point_indices(
        sampled_points,
        candidate_indices,
        int(n_points),
        center=np.zeros(3, dtype=np.float32),
    )
    return sampled_points[selected_indices], sampled_normals[selected_indices]


def _export_stl_point_cloud_ply(
    stl_path: Path,
    ply_path: Path,
    n_points: int = 8192,
    n_pre_points: int = 65536,
    mesh=None,
    sampled_points: Optional[np.ndarray] = None,
    sampled_normals: Optional[np.ndarray] = None,
) -> Path:
    if sampled_points is None or sampled_normals is None:
        if mesh is None:
            mesh = _load_trimesh_mesh(stl_path)
        sampled_points, sampled_normals = _sample_mesh_surface_points_with_normals(
            mesh,
            int(max(n_pre_points, n_points)),
        )
    points, normals = _select_point_cloud_from_samples(
        sampled_points,
        sampled_normals,
        n_points=int(n_points),
    )
    return _write_point_cloud_assets(points, ply_path, normals=normals)


def _sample_mesh_surface_points_with_normals(
    mesh,
    n_pre_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    import trimesh

    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

    sampled_points, face_indices = trimesh.sample.sample_surface(mesh, int(n_pre_points))
    sampled_points = np.asarray(sampled_points, dtype=np.float32)
    if getattr(mesh, "face_normals", None) is not None and len(mesh.face_normals) > 0:
        sampled_normals = np.asarray(mesh.face_normals[face_indices], dtype=np.float32)
    else:
        sampled_normals = np.zeros_like(sampled_points, dtype=np.float32)
    return sampled_points, sampled_normals


def _sample_point_indices(
    points: np.ndarray,
    candidate_indices: np.ndarray,
    target_count: int,
    center: np.ndarray,
) -> np.ndarray:
    if candidate_indices.size == 0:
        dists = np.linalg.norm(points - center[None, :], axis=1)
        nearest_idx = int(np.argmin(dists))
        return np.full((target_count,), nearest_idx, dtype=np.int64)

    if candidate_indices.size < target_count:
        repeats = (target_count + candidate_indices.size - 1) // candidate_indices.size
        tiled = np.tile(candidate_indices, repeats)[:target_count]
        return tiled.astype(np.int64)

    global _FPS_RUNTIME
    if _FPS_RUNTIME is None:
        import torch
        from pytorch3d.ops import sample_farthest_points
        _FPS_RUNTIME = (torch, sample_farthest_points)
    torch, sample_farthest_points = _FPS_RUNTIME

    selected_points = points[candidate_indices]
    selected_tensor = torch.from_numpy(np.asarray(selected_points, dtype=np.float32)).unsqueeze(0)
    with torch.no_grad():
        _, local_ids = sample_farthest_points(selected_tensor, K=int(target_count))
    return candidate_indices[local_ids[0].cpu().numpy()].astype(np.int64)


def _export_bbox_crop_point_cloud_from_samples(
    sampled_points: np.ndarray,
    sampled_normals: np.ndarray,
    bbox: Sequence[float],
    ply_path: Path,
    n_points: int = 8192,
    eps: float = 0.01,
    translation: Optional[Sequence[float]] = None,
    scale_factor: float = 1.0,
) -> Path:
    sampled_points = np.asarray(sampled_points, dtype=np.float32)
    sampled_normals = np.asarray(sampled_normals, dtype=np.float32)
    if sampled_normals.shape != sampled_points.shape:
        sampled_normals = np.zeros_like(sampled_points, dtype=np.float32)

    bbox_min = np.asarray(bbox[:3], dtype=np.float32)
    bbox_max = np.asarray(bbox[3:], dtype=np.float32)
    mask = np.all(
        (sampled_points >= bbox_min - float(eps)) &
        (sampled_points <= bbox_max + float(eps)),
        axis=1,
    )
    candidate_indices = np.nonzero(mask)[0]
    center = (bbox_min + bbox_max) / 2.0
    selected_indices = _sample_point_indices(sampled_points, candidate_indices, n_points, center=center)
    selected_points = sampled_points[selected_indices]
    selected_normals = sampled_normals[selected_indices]
    if translation is not None or not math.isclose(scale_factor, 1.0):
        if translation is None:
            translation = (0.0, 0.0, 0.0)
        selected_points = _apply_translation_scale(selected_points, translation, scale_factor)
    return _write_point_cloud_assets(selected_points, ply_path, normals=selected_normals)


def _sequence_indices_from_step_records(step_records: Sequence[dict]) -> list[int]:
    return sorted({
        int(seq_idx)
        for leaf in step_records
        for seq_idx in leaf.get("sequence_indices", [])
    })


def _filter_entity_groups_by_sequence_indices(
    entity_groups: Sequence[dict],
    sequence_indices: Sequence[int],
    renumber_indices: bool = False,
) -> list[dict]:
    ordered = [int(idx) for idx in sequence_indices]
    allowed = set(ordered)
    index_map = {idx: local_idx for local_idx, idx in enumerate(ordered)}
    filtered_groups = []

    for group in entity_groups:
        group_indices = [int(idx) for idx in group.get("indices", []) if int(idx) in allowed]
        if not group_indices:
            continue
        if renumber_indices:
            group_indices = [index_map[idx] for idx in group_indices]
        filtered_groups.append({
            "entity_id": group.get("entity_id"),
            "indices": group_indices,
            "operation": group.get("operation"),
        })

    return filtered_groups


def _step_payload_from_leaf_records(
    leaf_records: Sequence[dict],
    translation: Sequence[float],
    scale_factor: float,
    normalize: bool,
) -> list[tuple[str, list[float]]]:
    return [
        (
            leaf["step_name"],
            _bbox_array_to_list(
                _apply_translation_scale(leaf["bbox_np"], translation, scale_factor)
                if normalize else leaf["bbox_np"]
            ),
        )
        for leaf in sorted(leaf_records, key=lambda item: int(item["index"]))
    ]


def _workplane_from_shape(cq, shape):
    return cq.Workplane().add(shape)


def _entity_step_indices_map(leaf_records: Sequence[dict]) -> dict[int, set[int]]:
    entity_to_step_indices: dict[int, set[int]] = {}
    for leaf in leaf_records:
        entity_index = leaf.get("entity_index")
        if entity_index is None:
            continue
        entity_to_step_indices.setdefault(int(entity_index), set()).add(int(leaf["index"]))
    return entity_to_step_indices


def _transform_solid_for_part(cq, solid, translation: Sequence[float], scale_factor: float):
    shape = solid.val() if hasattr(solid, "val") else solid
    if translation is not None:
        shape = shape.translate(tuple(float(value) for value in translation))
    if not math.isclose(float(scale_factor), 1.0):
        shape = shape.scale(float(scale_factor))
    return _workplane_from_shape(cq, shape)


def _compose_part_solid_from_cached_artifacts(
    cq,
    selected_leaf_records: Sequence[dict],
    entity_groups: Sequence[dict],
    entity_artifacts: dict,
    entity_to_step_indices: dict[int, set[int]],
    translation: Sequence[float],
    scale_factor: float,
    normalize: bool,
):
    selected_indices = {int(leaf["index"]) for leaf in selected_leaf_records}

    items = []
    covered_indices: set[int] = set()
    for entity_index, all_step_indices in entity_to_step_indices.items():
        entity_selected = selected_indices & all_step_indices
        if not entity_selected or entity_selected != all_step_indices:
            continue
        entity_artifact = entity_artifacts.get(f"solid{entity_index}")
        if not entity_artifact or entity_artifact.get("solid") is None:
            continue
        operation = _bool_op_name_from_operation(entity_groups[entity_index].get("operation"))
        items.append((min(all_step_indices), operation, entity_artifact["solid"]))
        covered_indices.update(all_step_indices)

    result = None
    for leaf in sorted(selected_leaf_records, key=lambda item: int(item["index"])):
        if int(leaf["index"]) in covered_indices:
            continue
        items.append((int(leaf["index"]), leaf.get("bool_op", "union"), leaf.get("solid")))

    for _, operation, step_solid in sorted(items, key=lambda item: item[0]):
        if step_solid is None:
            return None
        if result is None:
            result = step_solid
            continue

        if operation == "cut":
            result = result.cut(step_solid)
        elif operation == "intersect":
            result = result.intersect(step_solid)
        else:
            result = result.union(step_solid)

    if result is None:
        return None

    if normalize:
        result = _transform_solid_for_part(cq, result, translation, scale_factor)
    return result


def _export_single_part_from_model_artifacts(
    cq,
    output_dir: Path,
    file_stem: str,
    full_code: str,
    model_artifacts: dict,
    leaf_records: Sequence[dict],
) -> dict:
    py_path = output_dir / f"{file_stem}.py"
    stl_path = output_dir / f"{file_stem}.stl"
    ply_path = output_dir / f"{file_stem}.ply"
    npy_path = output_dir / f"{file_stem}.npy"
    crop_ply_path = output_dir / f"{file_stem}_crop.ply"
    crop_npy_path = output_dir / f"{file_stem}_crop.npy"

    py_path.write_text(full_code, encoding="utf-8")
    if model_artifacts["valid"] and model_artifacts.get("solid") is not None:
        _export_cadquery_stl(cq, model_artifacts["solid"], stl_path)
        npy_path = _export_stl_point_cloud_ply(stl_path, ply_path)
        shutil.copyfile(ply_path, crop_ply_path)
        shutil.copyfile(npy_path, crop_npy_path)
        model_bbox = model_artifacts["bbox"]
    else:
        model_bbox = _bbox_array_to_list(
            _merge_bbox_arrays([np.asarray(leaf["bbox_np"], dtype=float) for leaf in leaf_records])
        )

    return {
        "local_bbox": model_bbox,
        "step_payload": _step_payload_from_leaf_records(
            leaf_records,
            translation=(0.0, 0.0, 0.0),
            scale_factor=1.0,
            normalize=False,
        ),
    }


def _export_step_part_like_output(
    cq,
    cad_seq: CADSequence,
    entity_groups: Sequence[dict],
    ordered_leaf_indices: Sequence[int],
    leaf_records_by_index: dict[int, dict],
    entity_artifacts: dict,
    entity_to_step_indices: dict[int, set[int]],
    output_dir: Path,
    file_stem: str,
    normalize: bool,
    scale: float,
) -> dict:
    ordered_leaf_indices = sorted(int(idx) for idx in ordered_leaf_indices)
    selected_leaf_records = [leaf_records_by_index[idx] for idx in ordered_leaf_indices]
    selected_sequence_indices = _sequence_indices_from_step_records(selected_leaf_records)
    base_bbox_inputs = [
        np.asarray(leaf["bbox_np"], dtype=float)
        for leaf in selected_leaf_records
        if leaf.get("bool_op") != "cut"
    ]
    if not base_bbox_inputs:
        base_bbox_inputs = [np.asarray(leaf["bbox_np"], dtype=float) for leaf in selected_leaf_records]
    global_part_bbox_np = _merge_bbox_arrays(base_bbox_inputs)

    part_translation = np.zeros(3, dtype=float)
    part_scale_factor = 1.0
    if normalize:
        part_translation, part_scale_factor = _normalization_transform_from_minmax_bbox(
            global_part_bbox_np,
            scale=scale,
        )

    part_extrudes = [deepcopy(cad_seq.seq[seq_idx]) for seq_idx in selected_sequence_indices]
    part_seq_bbox = np.stack([global_part_bbox_np[1], global_part_bbox_np[0]], axis=0)
    part_cad_seq = CADSequence(part_extrudes, part_seq_bbox)
    if normalize:
        part_cad_seq.transform(part_translation, part_scale_factor)

    local_entity_groups = _filter_entity_groups_by_sequence_indices(
        entity_groups,
        selected_sequence_indices,
        renumber_indices=True,
    )
    part_code = _sequence_to_code_verbose_by_entity_groups(part_cad_seq, local_entity_groups)

    part_py_path = output_dir / f"{file_stem}.py"
    part_stl_path = output_dir / f"{file_stem}.stl"
    part_ply_path = output_dir / f"{file_stem}.ply"
    part_npy_path = output_dir / f"{file_stem}.npy"
    part_py_path.write_text(part_code, encoding="utf-8")

    part_solid = _compose_part_solid_from_cached_artifacts(
        cq,
        selected_leaf_records,
        entity_groups,
        entity_artifacts,
        entity_to_step_indices,
        part_translation,
        part_scale_factor,
        normalize,
    )
    if part_solid is None:
        part_exec_code = _sequence_to_code_verbose_by_entity_groups(
            part_cad_seq,
            local_entity_groups,
            formatter=_fmt_full_precision,
            direct_solid=False,
        )
        part_exec_artifacts = _execute_code_artifacts(cq, part_exec_code, collect_volume=False)
        part_solid = part_exec_artifacts.get("solid") if part_exec_artifacts["valid"] else None

    if part_solid is not None:
        _export_cadquery_stl(cq, part_solid, part_stl_path)
        part_npy_path = _export_stl_point_cloud_ply(part_stl_path, part_ply_path)
        local_part_bbox = _bbox_array_to_list(_get_bbox_array_from_solid(part_solid))
    else:
        local_part_bbox = _bbox_array_to_list(
            _apply_translation_scale(global_part_bbox_np, part_translation, part_scale_factor)
            if normalize else global_part_bbox_np
        )

    step_payload = _step_payload_from_leaf_records(
        selected_leaf_records,
        part_translation,
        part_scale_factor,
        normalize,
    )

    return {
        "global_bbox": _bbox_array_to_list(global_part_bbox_np),
        "local_bbox": local_part_bbox,
        "step_payload": step_payload,
        "translation": part_translation,
        "scale_factor": part_scale_factor,
        "py_path": part_py_path,
        "stl_path": part_stl_path,
        "ply_path": part_ply_path,
        "npy_path": part_npy_path,
    }


def generate_full_output(
    json_path: str,
    output_dir: str,
    normalize: bool = True,
    scale: float = 1.0,
    quiet: bool = False,
) -> None:
    """
    Generate complete output structure for a JSON file.

    Creates:
        output_dir/
            {name}.py
            {name}.stl
            {name}.ply
            {name}.npy
            bbox.json
        output_dir.parent/
            {name}_1/
                {name}_1.py
                {name}_1.stl
                {name}_1.ply
                {name}_1.npy
                {name}_1_crop.ply
                {name}_1_crop.npy
                bbox.json
            {name}_2/
                ...

    Args:
        json_path: Path to DeepCAD JSON file
        output_dir: Output directory path
        normalize: Whether to normalize to the fixed symmetric range [-100, 100]
        scale: Uniform scale factor
        quiet: If True, suppress output messages
    """
    import cadquery as cq

    _SHAPE_DISTANCE_CACHE.clear()

    json_path = Path(json_path)
    output_dir = Path(output_dir)
    name = json_path.stem

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        for child in list(output_dir.iterdir()):
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    sibling_pattern = re.compile(rf"{re.escape(name)}_\d+")
    for sibling in output_dir.parent.iterdir():
        if sibling.is_dir() and sibling_pattern.fullmatch(sibling.name):
            shutil.rmtree(sibling)

    # Load and process JSON
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    cad_seq, entity_groups = _build_cad_sequence_with_entity_groups(payload)
    if normalize:
        _normalize_to_random_range(cad_seq, scale)
    elif not math.isclose(scale, 1.0):
        cad_seq.transform(0.0, scale)

    # Saved .py is rounded; all STL/export metrics use full precision code.
    full_code = _sequence_to_code_verbose_by_entity_groups(
        cad_seq,
        entity_groups,
    )
    exec_step_records: list[dict] = []
    full_exec_code = _sequence_to_code_verbose_by_entity_groups(
        cad_seq,
        entity_groups,
        formatter=_fmt_full_precision,
        step_records=exec_step_records,
        direct_solid=False,
    )

    full_step_names = [record["step_name"] for record in exec_step_records]
    full_entity_names = [f"solid{idx}" for idx in range(len(entity_groups))]
    model_artifacts = _execute_code_artifacts(
        cq,
        full_exec_code,
        step_names=full_step_names,
        entity_names=full_entity_names,
        collect_volume=False,
        collect_step_payload=False,
    )

    leaf_records = _build_step_leaf_records(
        exec_step_records,
        model_artifacts.get("step_artifacts", {}),
    )
    part_groups = _build_step_part_groups(cq, cad_seq, leaf_records)
    leaf_records_by_index = {int(leaf["index"]): leaf for leaf in leaf_records}
    entity_step_indices = _entity_step_indices_map(leaf_records)

    if len(part_groups) == 1:
        single_info = _export_single_part_from_model_artifacts(
            cq,
            output_dir,
            name,
            full_code,
            model_artifacts,
            leaf_records,
        )
        bbox_json = _serialize_part_bbox_json(
            model_bbox=single_info["local_bbox"],
            step_records=single_info["step_payload"],
        )
        bbox_path = output_dir / "bbox.json"
        with open(bbox_path, "w", encoding="utf-8") as f:
            json.dump(bbox_json, f, indent=2)

        if not quiet:
            print(f"[json2cadquery] Generated single-part output in {output_dir}")
            print(f"  - {name}.py")
            print(f"  - {name}.stl")
            print(f"  - {name}.ply")
            print(f"  - {name}.npy")
            print(f"  - {name}_crop.ply")
            print(f"  - {name}_crop.npy")
            print("  - bbox.json")
        return

    full_py_path = output_dir / f"{name}.py"
    full_stl_path = output_dir / f"{name}.stl"
    full_ply_path = output_dir / f"{name}.ply"
    full_py_path.write_text(full_code, encoding="utf-8")

    crop_sampled_points = None
    crop_sampled_normals = None
    if model_artifacts["valid"] and model_artifacts.get("solid") is not None:
        _export_cadquery_stl(cq, model_artifacts["solid"], full_stl_path)
        full_mesh = _load_trimesh_mesh(full_stl_path)
        crop_sampled_points, crop_sampled_normals = _sample_mesh_surface_points_with_normals(
            full_mesh,
            65536,
        )
        _export_stl_point_cloud_ply(
            full_stl_path,
            full_ply_path,
            sampled_points=crop_sampled_points,
            sampled_normals=crop_sampled_normals,
        )
        model_bbox = model_artifacts["bbox"]
    else:
        model_bbox = None

    part_payload_records: list[tuple[str, list[float]]] = []
    for part_idx, part in enumerate(part_groups, start=1):
        part_name = f"part_{part_idx}"
        folder_name = f"{name}_{part_idx}"
        part_output_dir = output_dir.parent / folder_name
        part_output_dir.mkdir(parents=True, exist_ok=True)

        part_info = _export_step_part_like_output(
            cq,
            cad_seq,
            entity_groups,
            part["leaf_indices"],
            leaf_records_by_index,
            model_artifacts.get("entity_artifacts", {}),
            entity_step_indices,
            part_output_dir,
            folder_name,
            normalize=normalize,
            scale=scale,
        )

        part_payload_records.append((part_name, part_info["global_bbox"]))
        if crop_sampled_points is not None and crop_sampled_normals is not None:
            part_crop_ply_path = part_output_dir / f"{folder_name}_crop.ply"
            _export_bbox_crop_point_cloud_from_samples(
                crop_sampled_points,
                crop_sampled_normals,
                part_info["global_bbox"],
                part_crop_ply_path,
                translation=part_info["translation"],
                scale_factor=part_info["scale_factor"],
            )

        part_json = _serialize_part_bbox_json(
            model_bbox=part_info["local_bbox"],
            step_records=part_info["step_payload"],
        )
        with open(part_output_dir / "bbox.json", "w", encoding="utf-8") as f:
            json.dump(part_json, f, indent=2)

        del part_info
        if part_idx % 20 == 0:
            gc.collect()

    if model_bbox is None:
        model_bbox = _bbox_array_to_list(
            _merge_bbox_arrays([_bbox_list_to_array(part_bbox) for _, part_bbox in part_payload_records])
        )

    bbox_json = _serialize_root_bbox_json(
        model_bbox=model_bbox,
        part_records=part_payload_records,
    )
    bbox_path = output_dir / "bbox.json"
    with open(bbox_path, "w", encoding="utf-8") as f:
        json.dump(bbox_json, f, indent=2)

    if not quiet:
        print(f"[json2cadquery] Generated output in {output_dir}")
        print(f"  - {name}.py")
        print(f"  - {name}.stl")
        print(f"  - {name}.ply")
        print(f"  - {name}.npy")
        print("  - bbox.json")
        print(f"  - sibling part dirs: {len(part_payload_records)}")


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = _parse_args(argv)

    if args.output_dir:
        # Full output mode
        generate_full_output(
            args.json,
            args.output_dir,
            normalize=not args.no_normalize,
            scale=args.scale,
        )
    else:
        # Simple mode
        code = json2cadquery(
            args.json,
            args.output,
            normalize=not args.no_normalize,
            by_entity=args.by_entity,
            scale=args.scale,
            verbose=args.verbose,
        )
        if args.output is None:
            print(code)
        else:
            print(f"[json2cadquery] CadQuery script written to {args.output}")


if __name__ == "__main__":
    main()
