"""Geometry-only wall detection for the raised top LiDAR."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin

import numpy as np


@dataclass(frozen=True)
class WallDetection:
    distance: float
    points: np.ndarray
    vertical_span: float
    cross_span: float
    score: float


def detect_transverse_wall(
    points: np.ndarray,
    axis_yaw: float,
    min_distance: float,
    max_distance: float,
    corridor_half_width: float,
    z_min: float,
    z_max: float,
    bin_size: float,
    min_points: int,
    min_vertical_span: float,
    min_cross_span: float,
    height_bin_size: float,
    min_occupied_height_bins: int,
    height_bin_min_points: int,
    low_height_limit: float,
    min_low_height_fraction: float,
    peak_neighbor_bins: int,
    min_peak_ratio: float,
) -> WallDetection | None:
    """Find the dominant vertical plane transverse to the insertion axis.

    Points are projected onto the configured bed axis. A wall produces many
    points in one narrow along-axis bin while spanning both height and the
    cross-axis corridor. Horizontal bed surfaces therefore fail the vertical
    span test.
    """
    cloud = np.asarray(points, dtype=np.float64)
    if cloud.ndim != 2 or cloud.shape[1] != 3 or len(cloud) == 0:
        return None

    axis = np.array([cos(axis_yaw), sin(axis_yaw)], dtype=np.float64)
    normal = np.array([-axis[1], axis[0]], dtype=np.float64)
    along = cloud[:, :2] @ axis
    cross = cloud[:, :2] @ normal
    valid = (
        np.isfinite(cloud).all(axis=1)
        & (along >= float(min_distance))
        & (along <= float(max_distance))
        & (np.abs(cross) <= float(corridor_half_width))
        & (cloud[:, 2] >= float(z_min))
        & (cloud[:, 2] <= float(z_max))
    )
    if not np.any(valid):
        return None

    source_indices = np.flatnonzero(valid)
    selected_along = along[valid]
    selected_cross = cross[valid]
    selected_z = cloud[valid, 2]
    width = max(float(bin_size), 1.0e-3)
    bins = np.floor(selected_along / width).astype(np.int64)
    unique_bins, bin_counts = np.unique(bins, return_counts=True)
    counts_by_bin = dict(zip(unique_bins.tolist(), bin_counts.tolist()))

    best: WallDetection | None = None
    for bin_id in unique_bins:
        local = bins == bin_id
        count = int(np.count_nonzero(local))
        if count < int(min_points):
            continue
        neighbor_count = max(2, int(peak_neighbor_bins))
        neighbor_values = []
        for offset in range(2, neighbor_count + 1):
            neighbor_values.append(counts_by_bin.get(int(bin_id) - offset, 0))
            neighbor_values.append(counts_by_bin.get(int(bin_id) + offset, 0))
        positive_neighbors = [value for value in neighbor_values if value > 0]
        if len(positive_neighbors) < 2:
            continue
        baseline = float(np.median(np.asarray(positive_neighbors)))
        if count < float(min_peak_ratio) * max(baseline, 1.0):
            continue
        z_low, z_high = np.percentile(selected_z[local], (10.0, 90.0))
        cross_low, cross_high = np.percentile(
            selected_cross[local], (10.0, 90.0)
        )
        vertical_span = float(z_high - z_low)
        cross_span = float(cross_high - cross_low)
        if vertical_span < float(min_vertical_span):
            continue
        if cross_span < float(min_cross_span):
            continue
        height_width = max(float(height_bin_size), 1.0e-3)
        height_bins = np.floor(
            (selected_z[local] - float(z_min)) / height_width
        ).astype(np.int64)
        height_counts = np.bincount(height_bins)
        occupied_height_bins = int(
            np.count_nonzero(height_counts >= int(height_bin_min_points))
        )
        if occupied_height_bins < int(min_occupied_height_bins):
            continue
        low_height_fraction = float(
            np.mean(selected_z[local] <= float(low_height_limit))
        )
        if low_height_fraction < float(min_low_height_fraction):
            continue
        score = float(count) * vertical_span * cross_span
        indices = source_indices[local]
        distance = float(np.median(selected_along[local]))
        candidate = WallDetection(
            distance=distance,
            points=cloud[indices].astype(np.float32, copy=True),
            vertical_span=vertical_span,
            cross_span=cross_span,
            score=score,
        )
        if best is None or candidate.score > best.score:
            best = candidate
    return best
