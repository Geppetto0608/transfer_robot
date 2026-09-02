from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from math import acos, atan2, exp, pi

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def wrap_angle(angle: float) -> float:
    return (angle + pi) % (2.0 * pi) - pi


def voxel_downsample(points: np.ndarray, leaf: float) -> np.ndarray:
    if len(points) == 0 or leaf <= 0.0:
        return points
    keys = np.floor(points / leaf).astype(np.int32)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    sums = np.zeros((inverse.max() + 1, 3), dtype=np.float64)
    counts = np.bincount(inverse)
    np.add.at(sums, inverse, points)
    return (sums / counts[:, None]).astype(np.float32)


def planar_downsample(points_xy: np.ndarray, leaf: float) -> np.ndarray:
    """Collapse repeated vertical returns into one representative XY cell.

    During insertion both the upper rail and thick lower frame can occupy the
    same XY footprint at many Z values. Keeping every return would bias a 2D
    fit toward whichever part happens to be tallest or densest.
    """
    if len(points_xy) == 0 or leaf <= 0.0:
        return points_xy
    keys = np.floor(points_xy / leaf).astype(np.int32)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    sums = np.zeros((inverse.max() + 1, 2), dtype=np.float64)
    counts = np.bincount(inverse)
    np.add.at(sums, inverse, points_xy)
    return (sums / counts[:, None]).astype(np.float64)


def detect_bed_top_rectangle(
    points_xy: np.ndarray,
    reference_entrance: np.ndarray,
    reference_yaw: float,
    expected_width: float = 0.90,
    width_tolerance: float = 0.20,
    expected_length: float = 2.04,
    length_tolerance: float = 0.45,
    min_observed_length: float = 0.55,
    min_points: int = 40,
    max_yaw_delta: float = 0.26,
    entrance_surface_offset: float = 0.0,
    projection_leaf: float = 0.04,
) -> BedDetection | None:
    """Fit a bed-sized top rectangle inside a foot-pose-seeded ROI.

    The merged cloud is height-cropped by the caller.  Repeated returns from
    all three LiDARs are collapsed into XY occupancy cells before fitting, so
    one sensor or a vertically thick surface cannot dominate the result.

    A complete four-edge rectangle is not required.  Both lateral extents and
    a useful longitudinal span must be visible; the known bed length and the
    previous foot/top entrance pose are then used to infer the entrance.
    """
    if len(points_xy) < min_points:
        return None
    occupied = planar_downsample(points_xy.astype(np.float64), projection_leaf)
    if len(occupied) < min_points:
        return None

    components = euclidean_clusters(
        occupied,
        tolerance=max(0.06, 2.2 * projection_leaf),
        min_points=min_points,
        max_points=len(occupied),
    )
    if not components:
        return None

    reference_entrance = np.asarray(reference_entrance, dtype=np.float64)
    best: BedDetection | None = None
    best_cost = float("inf")
    yaw_steps = np.linspace(-max_yaw_delta, max_yaw_delta, 31)
    for indices in components:
        component = occupied[indices]
        for yaw_delta in yaw_steps:
            yaw = wrap_angle(reference_yaw + float(yaw_delta))
            axis = np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float64)
            normal = np.array([-axis[1], axis[0]], dtype=np.float64)
            relative = component - reference_entrance
            along = relative @ axis
            cross = relative @ normal

            along_low, along_high = np.percentile(along, (2.0, 98.0))
            cross_low, cross_high = np.percentile(cross, (2.0, 98.0))
            observed_width = float(cross_high - cross_low)
            observed_length = float(along_high - along_low)
            width_error = abs(observed_width - expected_width)
            if width_error > width_tolerance:
                continue
            if observed_length < min_observed_length:
                continue
            if observed_length > expected_length + length_tolerance:
                continue

            # The visible near edge and the visible far edge each provide an
            # entrance hypothesis. Select the one continuous with the preceding
            # foot/top pose. ``entrance_surface_offset`` handles a mattress whose
            # top begins slightly behind the physical foot-board.
            from_near = float(along_low - entrance_surface_offset)
            from_far = float(along_high - expected_length)
            entrance_along = min((from_near, from_far), key=abs)
            entrance_cross = 0.5 * float(cross_low + cross_high)
            entrance = (
                reference_entrance
                + entrance_along * axis
                + entrance_cross * normal
            )
            far = entrance + expected_length * axis

            width_cost = width_error / max(width_tolerance, 1.0e-6)
            yaw_cost = abs(float(yaw_delta)) / max(max_yaw_delta, 1.0e-6)
            length_coverage = min(
                1.0, observed_length / max(expected_length, 1.0e-6)
            )
            point_coverage = min(
                1.0, len(component) / max(2.0 * min_points, 1.0)
            )
            cost = width_cost + 0.35 * yaw_cost + 0.35 * (1.0 - length_coverage)
            confidence = float(
                np.clip(
                    0.95
                    * exp(-0.45 * cost)
                    * (0.75 + 0.25 * point_coverage),
                    0.0,
                    1.0,
                )
            )
            if cost >= best_cost:
                continue
            half_width = 0.5 * observed_width * normal
            best_cost = cost
            best = BedDetection(
                entrance_center=entrance.astype(np.float64),
                far_center=far.astype(np.float64),
                yaw=yaw,
                confidence=confidence,
                corners=np.vstack(
                    (
                        entrance + half_width,
                        entrance - half_width,
                        far + half_width,
                        far - half_width,
                    )
                ).astype(np.float64),
                width=observed_width,
                length=expected_length,
                evidence_points=component.astype(np.float64, copy=True),
            )
    return best


def radius_outlier_filter(points: np.ndarray, radius: float, min_neighbors: int) -> np.ndarray:
    if len(points) == 0 or min_neighbors <= 1:
        return points
    tree = cKDTree(points)
    neighbors = tree.query_ball_point(points, radius)
    keep = np.fromiter((len(items) >= min_neighbors for items in neighbors), dtype=bool)
    return points[keep]


def euclidean_clusters(
    points_xy: np.ndarray,
    tolerance: float,
    min_points: int,
    max_points: int,
) -> list[np.ndarray]:
    if len(points_xy) == 0:
        return []
    tree = cKDTree(points_xy)
    pairs = tree.query_pairs(tolerance, output_type="ndarray")
    count = len(points_xy)
    if len(pairs) == 0:
        labels = np.arange(count, dtype=np.int32)
    else:
        rows = np.concatenate((pairs[:, 0], pairs[:, 1]))
        cols = np.concatenate((pairs[:, 1], pairs[:, 0]))
        graph = coo_matrix(
            (np.ones(len(rows), dtype=np.uint8), (rows, cols)),
            shape=(count, count),
        )
        _, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels, minlength=int(labels.max()) + 1)
    return [
        np.flatnonzero(labels == label).astype(np.int32)
        for label, size in enumerate(sizes)
        if min_points <= int(size) <= max_points
    ]


@dataclass(frozen=True)
class ClusterFeature:
    center: np.ndarray
    size: np.ndarray
    count: int


@dataclass(frozen=True)
class BedDetection:
    entrance_center: np.ndarray
    far_center: np.ndarray
    yaw: float
    confidence: float
    corners: np.ndarray
    width: float
    length: float
    # XY evidence supporting the selected RANSAC geometry. Older leg-based
    # helper functions leave this empty; active FOOT/TOP/SIDE detectors fill it.
    evidence_points: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.float64)
    )


@dataclass(frozen=True)
class LineModel:
    center: np.ndarray
    direction: np.ndarray
    endpoints: np.ndarray
    length: float
    residual: float
    inlier_ratio: float
    inliers: np.ndarray


def _ransac_transverse_lines(
    points_xy: np.ndarray,
    distance_threshold: float,
    min_points: int,
    line_axis: str,
    min_abs_direction: float,
    iterations: int = 100,
    max_models: int = 4,
) -> list[LineModel]:
    """Extract transverse line models, removing each model before the next."""
    if len(points_xy) < min_points:
        return []
    remaining = points_xy.astype(np.float64, copy=True)
    rng = np.random.default_rng(31 + len(points_xy))
    models: list[LineModel] = []
    axis_index = 0 if line_axis.upper() == "X" else 1

    for _ in range(max_models):
        if len(remaining) < min_points:
            break
        best_mask = None
        best_count = 0
        best_span = 0.0
        for _ in range(iterations):
            i, j = rng.choice(len(remaining), size=2, replace=False)
            vector = remaining[j] - remaining[i]
            norm = float(np.linalg.norm(vector))
            if norm < 0.10:
                continue
            direction = vector / norm
            if abs(float(direction[axis_index])) < min_abs_direction:
                continue
            normal = np.array([-direction[1], direction[0]])
            distances = np.abs((remaining - remaining[i]) @ normal)
            mask = distances <= distance_threshold
            count = int(mask.sum())
            if count < min_points:
                continue
            span = float(np.ptp((remaining[mask] - remaining[i]) @ direction))
            if count > best_count or (count == best_count and span > best_span):
                best_mask, best_count, best_span = mask, count, span

        if best_mask is None:
            break
        inliers = remaining[best_mask]
        center_mean = inliers.mean(axis=0)
        covariance = np.cov((inliers - center_mean).T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        direction = eigenvectors[:, int(np.argmax(eigenvalues))]
        if direction[axis_index] < 0.0:
            direction = -direction
        if abs(float(direction[axis_index])) < min_abs_direction:
            remaining = remaining[~best_mask]
            continue
        normal = np.array([-direction[1], direction[0]])
        projection = (inliers - center_mean) @ direction
        low, high = np.percentile(projection, (5.0, 95.0))
        endpoints = np.vstack((center_mean + low * direction, center_mean + high * direction))
        line_center = 0.5 * (endpoints[0] + endpoints[1])
        residual = float(np.mean(np.abs((inliers - center_mean) @ normal)))
        models.append(
            LineModel(
                center=line_center,
                direction=direction,
                endpoints=endpoints,
                length=float(high - low),
                residual=residual,
                inlier_ratio=float(len(inliers) / len(points_xy)),
                inliers=inliers.astype(np.float64, copy=True),
            )
        )
        remaining = remaining[~best_mask]
    return models


def _ransac_longitudinal_lines(
    points_xy: np.ndarray,
    distance_threshold: float,
    min_points: int,
    min_abs_direction_x: float,
    iterations: int = 120,
    max_models: int = 6,
) -> list[LineModel]:
    """Extract bed-length line models whose direction is mostly base X."""
    if len(points_xy) < min_points:
        return []
    remaining = points_xy.astype(np.float64, copy=True)
    rng = np.random.default_rng(73 + len(points_xy))
    models: list[LineModel] = []

    for _ in range(max_models):
        if len(remaining) < min_points:
            break
        best_mask = None
        best_count = 0
        best_span = 0.0
        for _ in range(iterations):
            i, j = rng.choice(len(remaining), size=2, replace=False)
            vector = remaining[j] - remaining[i]
            norm = float(np.linalg.norm(vector))
            if norm < 0.10:
                continue
            direction = vector / norm
            if abs(float(direction[0])) < min_abs_direction_x:
                continue
            normal = np.array([-direction[1], direction[0]])
            distances = np.abs((remaining - remaining[i]) @ normal)
            mask = distances <= distance_threshold
            count = int(mask.sum())
            if count < min_points:
                continue
            span = float(np.ptp((remaining[mask] - remaining[i]) @ direction))
            if count > best_count or (count == best_count and span > best_span):
                best_mask, best_count, best_span = mask, count, span

        if best_mask is None:
            break
        inliers = remaining[best_mask]
        center_mean = inliers.mean(axis=0)
        covariance = np.cov((inliers - center_mean).T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        direction = eigenvectors[:, int(np.argmax(eigenvalues))]
        if direction[0] < 0.0:
            direction = -direction
        if abs(float(direction[0])) < min_abs_direction_x:
            remaining = remaining[~best_mask]
            continue
        normal = np.array([-direction[1], direction[0]])
        projection = (inliers - center_mean) @ direction
        low, high = np.percentile(projection, (5.0, 95.0))
        endpoints = np.vstack((center_mean + low * direction, center_mean + high * direction))
        residual = float(np.mean(np.abs((inliers - center_mean) @ normal)))
        models.append(
            LineModel(
                center=0.5 * (endpoints[0] + endpoints[1]),
                direction=direction,
                endpoints=endpoints,
                length=float(high - low),
                residual=residual,
                inlier_ratio=float(len(inliers) / len(points_xy)),
                inliers=inliers.astype(np.float64, copy=True),
            )
        )
        remaining = remaining[~best_mask]
    return models


def detect_bed_side_lines(
    points_xy: np.ndarray,
    clusters: list[np.ndarray],
    expected_width: float = 0.90,
    width_tolerance: float = 0.22,
    bed_length: float = 2.04,
    distance_threshold: float = 0.06,
    min_points: int = 20,
    min_observed_length: float = 0.75,
    min_abs_direction_x: float = 0.70,
    parallel_tolerance_deg: float = 12.0,
    bin_width: float = 0.05,
    bin_min_points: int = 8,
    boundary_low_percentile: float = 3.0,
    boundary_high_percentile: float = 97.0,
    boundary_distance_threshold: float = 0.035,
    boundary_min_bins: int = 6,
    symmetry_weight: float = 2.0,
    envelope_preference: float = 0.35,
    tracking_position_weight: float = 1.2,
    tracking_reference: np.ndarray | None = None,
    tracking_yaw: float | None = None,
) -> BedDetection | None:
    """Detect the two long bed boundaries during insertion.

    Upper rails, thick lower frames, and horizontal surfaces can project to a
    filled XY rectangle. Running RANSAC on every point can then pair an inner
    face on one side with an outer face on the other. Prefer the lower/upper Y
    envelope in short X bins, which represents the complete structure's outer
    boundaries. Individual longitudinal line pairs remain as a fallback for a
    sparse view where the two rails are separate clusters.
    """
    candidate_pairs: list[tuple[LineModel, LineModel, bool]] = []
    individual_lines: list[LineModel] = []
    for indices in clusters:
        cloud = points_xy[indices]
        individual_lines.extend(
            _ransac_longitudinal_lines(
                cloud,
                distance_threshold=distance_threshold,
                min_points=min_points,
                min_abs_direction_x=min_abs_direction_x,
            )
        )

        # A single Euclidean cluster commonly contains the complete bed-top
        # surface. Extract its Y envelope before line fitting.
        if len(cloud) < max(20, min_points) or float(np.ptp(cloud[:, 0])) < 0.35:
            continue
        keys = np.floor(cloud[:, 0] / bin_width).astype(np.int32)
        lower: list[list[float]] = []
        upper: list[list[float]] = []
        for key in np.unique(keys):
            section = cloud[keys == key]
            if len(section) < bin_min_points:
                continue
            x = float(np.median(section[:, 0]))
            lower.append([x, float(np.percentile(section[:, 1], boundary_low_percentile))])
            upper.append([x, float(np.percentile(section[:, 1], boundary_high_percentile))])
        if len(lower) < boundary_min_bins:
            continue
        boundary_min_points = max(boundary_min_bins, min_points // 4)
        boundary_threshold = min(boundary_distance_threshold, distance_threshold)
        lower_models = _ransac_longitudinal_lines(
            np.asarray(lower, dtype=np.float64),
            distance_threshold=boundary_threshold,
            min_points=boundary_min_points,
            min_abs_direction_x=min_abs_direction_x,
            iterations=160,
            max_models=1,
        )
        upper_models = _ransac_longitudinal_lines(
            np.asarray(upper, dtype=np.float64),
            distance_threshold=boundary_threshold,
            min_points=boundary_min_points,
            min_abs_direction_x=min_abs_direction_x,
            iterations=160,
            max_models=1,
        )
        if lower_models and upper_models:
            candidate_pairs.append((lower_models[0], upper_models[0], True))

    candidate_pairs.extend(
        (first, second, False)
        for first, second in combinations(individual_lines, 2)
    )

    parallel_limit = np.deg2rad(parallel_tolerance_deg)
    best: BedDetection | None = None
    best_cost = float("inf")
    for first, second, is_envelope_pair in candidate_pairs:
        if first.length < min_observed_length or second.length < min_observed_length:
            continue
        direction_a = first.direction.copy()
        direction_b = second.direction.copy()
        if np.dot(direction_a, direction_b) < 0.0:
            direction_b = -direction_b
        parallel_error = acos(float(np.clip(np.dot(direction_a, direction_b), -1.0, 1.0)))
        if parallel_error > parallel_limit:
            continue

        # Perpendicular feet from LiDAR/base origin to both fitted X lines.
        foot_a = first.center - float(np.dot(first.center, direction_a)) * direction_a
        foot_b = second.center - float(np.dot(second.center, direction_b)) * direction_b
        cross_line = foot_b - foot_a
        if cross_line[1] < 0.0:
            first, second = second, first
            direction_a, direction_b = direction_b, direction_a
            foot_a, foot_b = foot_b, foot_a
            cross_line = -cross_line
        separation = float(np.linalg.norm(cross_line))
        width_error = abs(separation - expected_width)
        if width_error > width_tolerance:
            continue

        # For a correctly aligned bed the foot-to-foot line points along +Y.
        yaw = atan2(float(-cross_line[0]), float(cross_line[1]))
        bed_axis = np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float64)
        normal = np.array([-bed_axis[1], bed_axis[0]])

        endpoints = np.vstack((first.endpoints, second.endpoints))
        along = endpoints @ bed_axis
        observed_length = float(np.percentile(along, 95.0) - np.percentile(along, 5.0))
        if observed_length < min_observed_length:
            continue
        center_along = 0.5 * (float(np.percentile(along, 5.0)) + float(np.percentile(along, 95.0)))
        feet_midpoint = 0.5 * (foot_a + foot_b)
        center_side = float(np.dot(feet_midpoint, normal))
        bed_center = center_along * bed_axis + center_side * normal
        entrance_center = bed_center - 0.5 * bed_length * bed_axis
        far_center = bed_center + 0.5 * bed_length * bed_axis
        residual_error = 0.5 * (first.residual + second.residual) / max(distance_threshold, 1.0e-6)
        length_error = abs(observed_length - bed_length) / max(0.5 * bed_length, 1.0e-6)
        cost = (
            width_error / max(width_tolerance, 1.0e-6)
            + parallel_error / max(parallel_limit, 1.0e-6)
            + 0.4 * residual_error
            + 0.7 * length_error
        )
        # Prefer pairs that place the LiDAR between the two bed boundaries.
        side_a = float(np.dot(foot_a, normal))
        side_b = float(np.dot(foot_b, normal))
        if side_a * side_b > 0.0:
            cost += 2.0
        cost += symmetry_weight * abs(center_side) / max(
            0.5 * expected_width, 1.0e-6
        )
        if not is_envelope_pair:
            cost += envelope_preference
        if tracking_reference is not None:
            # The entrance inferred from side boundaries must remain continuous
            # with the foot-board pose. Using full XY distance prevents a
            # geometrically plausible but longitudinally unrelated frame from
            # taking over during or immediately after the handoff.
            cost += tracking_position_weight * float(
                np.linalg.norm(entrance_center - tracking_reference)
            )
        if tracking_yaw is not None:
            cost += 2.0 * abs(wrap_angle(yaw - tracking_yaw))

        confidence = float(np.clip(0.95 * exp(-0.30 * cost), 0.0, 1.0))
        if cost < best_cost:
            half_width = 0.5 * separation * normal
            best_cost = cost
            best = BedDetection(
                entrance_center=entrance_center.astype(np.float64),
                far_center=far_center.astype(np.float64),
                yaw=yaw,
                confidence=confidence,
                corners=np.vstack(
                    (
                        entrance_center + half_width,
                        entrance_center - half_width,
                        far_center + half_width,
                        far_center - half_width,
                    )
                ).astype(np.float64),
                width=separation,
                length=bed_length,
                evidence_points=np.vstack((first.inliers, second.inliers)).astype(np.float64),
            )
    return best


def detect_bed_top_outlines(
    points_xy: np.ndarray,
    reference_entrance: np.ndarray,
    reference_yaw: float,
    expected_width: float = 0.90,
    width_tolerance: float = 0.20,
    bed_length: float = 2.04,
    distance_threshold: float = 0.06,
    min_points: int = 20,
    min_observed_length: float = 0.55,
    min_abs_direction: float = 0.82,
    parallel_tolerance_deg: float = 12.0,
    tracking_position_weight: float = 1.2,
) -> BedDetection | None:
    """Estimate a bed pose from the two visible outlines of its top surface.

    Unlike :func:`detect_bed_top_rectangle`, this does not score a filled XY
    patch by its bounding box.  It first extracts each long top outline with
    RANSAC, then accepts only a parallel pair whose separation is the known
    bed width.  This is robust when the merged scanners fill only part of the
    mattress or include a thick lower structure in the height crop.
    """
    if len(points_xy) < min_points:
        return None

    axis = np.array([np.cos(reference_yaw), np.sin(reference_yaw)], dtype=np.float64)
    normal = np.array([-axis[1], axis[0]], dtype=np.float64)
    basis = np.vstack((axis, normal))
    local_points = planar_downsample(points_xy.astype(np.float64) @ basis.T, 0.04)
    if len(local_points) < min_points:
        return None
    clusters = euclidean_clusters(
        local_points,
        tolerance=0.14,
        min_points=max(8, min_points // 3),
        max_points=max(len(local_points), min_points),
    )
    if not clusters:
        return None

    local_reference = np.asarray(reference_entrance, dtype=np.float64) @ basis.T
    local = detect_bed_side_lines(
        local_points,
        clusters,
        expected_width=expected_width,
        width_tolerance=width_tolerance,
        bed_length=bed_length,
        distance_threshold=distance_threshold,
        min_points=min_points,
        min_observed_length=min_observed_length,
        min_abs_direction_x=min_abs_direction,
        parallel_tolerance_deg=parallel_tolerance_deg,
        bin_min_points=4,
        boundary_min_bins=5,
        tracking_position_weight=tracking_position_weight,
        tracking_reference=local_reference,
        tracking_yaw=0.0,
    )
    if local is None:
        return None

    def to_base(values: np.ndarray) -> np.ndarray:
        return values @ basis

    return BedDetection(
        entrance_center=to_base(local.entrance_center),
        far_center=to_base(local.far_center),
        yaw=wrap_angle(local.yaw + reference_yaw),
        confidence=local.confidence,
        corners=to_base(local.corners),
        width=local.width,
        length=local.length,
        evidence_points=to_base(local.evidence_points),
    )


def apply_bed_pose_calibration(
    detection: BedDetection,
    along_offset: float,
    cross_offset: float,
    yaw_offset: float,
) -> BedDetection:
    """Apply a rigid FOOT-to-TOP model calibration to a detection.

    The offsets are stored in the raw TOP model's bed frame, so they continue
    to rotate correctly if the robot makes a small yaw correction during
    insertion.  Evidence points remain raw sensor evidence; only the inferred
    bed geometry is calibrated.
    """
    raw_axis = np.array(
        [np.cos(detection.yaw), np.sin(detection.yaw)], dtype=np.float64
    )
    raw_normal = np.array([-raw_axis[1], raw_axis[0]], dtype=np.float64)
    entrance = (
        detection.entrance_center
        + float(along_offset) * raw_axis
        + float(cross_offset) * raw_normal
    )
    yaw = wrap_angle(detection.yaw + float(yaw_offset))
    axis = np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float64)
    normal = np.array([-axis[1], axis[0]], dtype=np.float64)
    far = entrance + detection.length * axis
    half_width = 0.5 * detection.width * normal
    return BedDetection(
        entrance_center=entrance.astype(np.float64),
        far_center=far.astype(np.float64),
        yaw=yaw,
        confidence=detection.confidence,
        corners=np.vstack(
            (
                entrance + half_width,
                entrance - half_width,
                far + half_width,
                far - half_width,
            )
        ).astype(np.float64),
        width=detection.width,
        length=detection.length,
        evidence_points=detection.evidence_points,
    )


def detect_bed_foot_line(
    points_xy: np.ndarray,
    clusters: list[np.ndarray],
    expected_width: float = 0.95,
    width_tolerance: float = 0.30,
    bed_length: float = 2.04,
    distance_threshold: float = 0.06,
    min_points: int = 20,
    min_observed_length: float = 0.55,
    line_axis: str = "Y",
    min_abs_direction: float = 0.70,
    tracking_reference: np.ndarray | None = None,
    tracking_yaw: float | None = None,
    tracking_position_weight: float = 0.8,
    tracking_yaw_weight: float = 2.5,
    bed_axis_sign: float = 1.0,
) -> BedDetection | None:
    """Detect the bed foot-board/frame as a transverse RANSAC line.

    No bed-leg candidates are used. ``line_axis`` selects whether the observed
    bed-width line is expected along base X or Y. The bed axis is the positive
    normal of that selected axis, and the closest validated transverse frame
    is treated as the docking entrance.
    """
    line_axis = line_axis.strip().upper()
    if line_axis not in ("X", "Y"):
        raise ValueError(f"line_axis must be X or Y, got '{line_axis}'")
    line_axis_index = 0 if line_axis == "X" else 1
    bed_axis_index = 1 - line_axis_index
    candidates: list[LineModel] = []
    for indices in clusters:
        candidates.extend(
            _ransac_transverse_lines(
                points_xy[indices],
                distance_threshold=distance_threshold,
                min_points=min_points,
                line_axis=line_axis,
                min_abs_direction=min_abs_direction,
            )
        )

    best: BedDetection | None = None
    best_cost = float("inf")
    for line in candidates:
        if line.length < min_observed_length:
            continue
        if abs(line.length - expected_width) > width_tolerance:
            continue
        if line.residual > distance_threshold:
            continue

        bed_axis = np.array([line.direction[1], -line.direction[0]], dtype=np.float64)
        desired_axis_sign = 1.0 if bed_axis_sign >= 0.0 else -1.0
        if bed_axis[bed_axis_index] * desired_axis_sign < 0.0:
            bed_axis = -bed_axis
        yaw = atan2(float(bed_axis[1]), float(bed_axis[0]))
        far_center = line.center + bed_length * bed_axis
        far_endpoints = line.endpoints + bed_length * bed_axis

        width_error = abs(line.length - expected_width) / max(width_tolerance, 1.0e-6)
        residual_error = line.residual / max(distance_threshold, 1.0e-6)
        orientation_error = abs(float(line.direction[bed_axis_index])) / max(
            1.0e-6, 1.0 - min_abs_direction
        )
        if tracking_reference is None:
            # The foot-board is the first valid transverse bed-width line seen
            # from the docking entrance. A stronger distance term prevents the
            # far board or an ROI-boundary line from winning on width alone.
            association_cost = 1.0 * float(np.linalg.norm(line.center))
        else:
            association_cost = tracking_position_weight * float(
                np.linalg.norm(line.center - tracking_reference)
            )
        if tracking_yaw is not None:
            association_cost += tracking_yaw_weight * abs(
                wrap_angle(yaw - tracking_yaw)
            )
        cost = width_error + 0.5 * residual_error + 0.25 * orientation_error + association_cost
        confidence = float(np.clip(0.95 * exp(-0.35 * cost), 0.0, 1.0))
        if cost < best_cost:
            best_cost = cost
            best = BedDetection(
                entrance_center=line.center.astype(np.float64),
                far_center=far_center,
                yaw=yaw,
                confidence=confidence,
                corners=np.vstack((line.endpoints, far_endpoints)).astype(np.float64),
                width=line.length,
                length=bed_length,
                evidence_points=line.inliers.astype(np.float64, copy=True),
            )
    return best


def cluster_features(points: np.ndarray, clusters: list[np.ndarray]) -> list[ClusterFeature]:
    features: list[ClusterFeature] = []
    for indices in clusters:
        cloud = points[indices]
        low = np.percentile(cloud, 10.0, axis=0)
        high = np.percentile(cloud, 90.0, axis=0)
        features.append(
            ClusterFeature(
                center=np.median(cloud, axis=0),
                size=high - low,
                count=len(indices),
            )
        )
    return features


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / max(norm, 1.0e-9)


def _unsigned_angle(a: np.ndarray, b: np.ndarray) -> float:
    return acos(float(np.clip(abs(np.dot(_unit(a), _unit(b))), -1.0, 1.0)))


def detect_bed_rectangle(
    centers_xy: np.ndarray,
    expected_width: float = 0.68,
    expected_length: float = 1.56,
    width_tolerance: float = 0.16,
    length_tolerance: float = 0.40,
    parallel_tolerance_deg: float = 12.0,
    orthogonal_tolerance_deg: float = 15.0,
    tracking_reference: np.ndarray | None = None,
) -> BedDetection | None:
    """Find four bed-leg centers forming the expected rectangle.

    The nearer transverse pair is defined as the bed entrance.  The pose yaw
    points from that entrance toward the far pair, which removes the 180-degree
    line ambiguity for a robot arriving in front of the bed.
    """
    if len(centers_xy) < 4:
        return None

    parallel_limit = np.deg2rad(parallel_tolerance_deg)
    orthogonal_limit = np.deg2rad(orthogonal_tolerance_deg)
    partitions = (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2)))
    best: BedDetection | None = None
    best_cost = float("inf")

    # Limiting candidates prevents a crowded scene from causing combinatorial load.
    nearest = np.argsort(np.linalg.norm(centers_xy, axis=1))[:12]
    for selected in combinations(nearest.tolist(), 4):
        points = centers_xy[list(selected)]
        for first_pair, second_pair in partitions:
            a, b = points[list(first_pair)]
            c, d = points[list(second_pair)]
            v1 = b - a
            v2 = d - c
            width1 = float(np.linalg.norm(v1))
            width2 = float(np.linalg.norm(v2))
            if abs(width1 - expected_width) > width_tolerance:
                continue
            if abs(width2 - expected_width) > width_tolerance:
                continue

            center1 = 0.5 * (a + b)
            center2 = 0.5 * (c + d)
            axis = center2 - center1
            length = float(np.linalg.norm(axis))
            if abs(length - expected_length) > length_tolerance:
                continue

            parallel_error = _unsigned_angle(v1, v2)
            orthogonal_error = abs(pi / 2.0 - _unsigned_angle(v1, axis))
            if parallel_error > parallel_limit or orthogonal_error > orthogonal_limit:
                continue

            # Orient both transverse pairs in the same direction, then compare side lengths.
            if np.dot(v1, v2) < 0.0:
                c, d = d, c
                v2 = -v2
            side1 = float(np.linalg.norm(c - a))
            side2 = float(np.linalg.norm(d - b))
            symmetry_error = abs(side1 - side2)
            if symmetry_error > 0.20:
                continue

            width_error = 0.5 * (abs(width1 - expected_width) + abs(width2 - expected_width))
            length_error = abs(length - expected_length)
            cost = (
                width_error / max(width_tolerance, 1.0e-6)
                + length_error / max(length_tolerance, 1.0e-6)
                + parallel_error / max(parallel_limit, 1.0e-6)
                + orthogonal_error / max(orthogonal_limit, 1.0e-6)
                + symmetry_error / 0.20
            )

            if tracking_reference is not None:
                swap_pairs = np.linalg.norm(center2 - tracking_reference) < np.linalg.norm(
                    center1 - tracking_reference
                )
            else:
                swap_pairs = np.linalg.norm(center2) < np.linalg.norm(center1)
            if swap_pairs:
                center1, center2 = center2, center1
                a, b, c, d = c, d, a, b
                axis = center2 - center1

            if tracking_reference is not None:
                cost += 0.25 * float(np.linalg.norm(center1 - tracking_reference))

            confidence = float(np.clip(exp(-0.35 * cost), 0.0, 1.0))
            yaw = atan2(float(axis[1]), float(axis[0]))
            if cost < best_cost:
                best_cost = cost
                best = BedDetection(
                    entrance_center=center1.astype(np.float64),
                    far_center=center2.astype(np.float64),
                    yaw=yaw,
                    confidence=confidence,
                    corners=np.vstack((a, b, c, d)).astype(np.float64),
                    width=0.5 * (width1 + width2),
                    length=length,
                )
    return best


def detect_bed_pair(
    centers_xy: np.ndarray,
    expected_width: float = 0.68,
    expected_length: float = 1.56,
    width_tolerance: float = 0.16,
    tracking_reference: np.ndarray | None = None,
) -> BedDetection | None:
    """Infer a bed pose from the visible entrance leg pair.

    Rear legs are commonly occluded in a frontal docking view.  This fallback
    selects the transverse pair whose spacing best matches the bed width.  The
    longitudinal axis is the pair's perpendicular facing robot-forward (+X).
    Its confidence is intentionally capped below a four-leg rectangle match.
    """
    if len(centers_xy) < 2:
        return None
    best = None
    best_cost = float("inf")
    nearest = np.argsort(np.linalg.norm(centers_xy, axis=1))[:12]
    for i, j in combinations(nearest.tolist(), 2):
        a = centers_xy[i].astype(np.float64)
        b = centers_xy[j].astype(np.float64)
        transverse = b - a
        width = float(np.linalg.norm(transverse))
        width_error = abs(width - expected_width)
        if width_error > width_tolerance:
            continue
        center = 0.5 * (a + b)
        direction = _unit(np.array([-transverse[1], transverse[0]], dtype=np.float64))
        if direction[0] < 0.0:
            direction = -direction
        # Once acquired, keep following the same physical leg pair. This
        # prevents a 1.56 m target jump when the AMR crosses the entrance.
        if tracking_reference is None:
            association_cost = 0.01 * float(np.linalg.norm(center))
        else:
            association_cost = 2.0 * float(np.linalg.norm(center - tracking_reference))
        cost = width_error / max(width_tolerance, 1.0e-6) + association_cost
        if cost < best_cost:
            far_a = a + expected_length * direction
            far_b = b + expected_length * direction
            confidence = float(0.68 * exp(-0.8 * width_error / max(width_tolerance, 1.0e-6)))
            best_cost = cost
            best = BedDetection(
                entrance_center=center,
                far_center=center + expected_length * direction,
                yaw=atan2(float(direction[1]), float(direction[0])),
                confidence=confidence,
                corners=np.vstack((a, b, far_a, far_b)),
                width=width,
                length=expected_length,
            )
    return best
