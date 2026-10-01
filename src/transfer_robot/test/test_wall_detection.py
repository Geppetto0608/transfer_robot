from math import pi

import numpy as np

from transfer_robot.wall_detection import detect_transverse_wall


PARAMS = dict(
    axis_yaw=-pi / 2.0,
    min_distance=0.2,
    max_distance=4.3,
    corridor_half_width=1.5,
    z_min=0.15,
    z_max=2.5,
    bin_size=0.05,
    min_points=100,
    min_vertical_span=0.8,
    min_cross_span=0.8,
    height_bin_size=0.1,
    min_occupied_height_bins=6,
    height_bin_min_points=5,
    low_height_limit=1.8,
    min_low_height_fraction=0.3,
    peak_neighbor_bins=3,
    min_peak_ratio=1.5,
)


def test_detects_dense_vertical_wall_transverse_to_bed_axis():
    rng = np.random.default_rng(7)
    wall = np.column_stack(
        (
            rng.uniform(-1.0, 1.0, 1500),
            rng.normal(-1.03, 0.008, 1500),
            rng.uniform(0.25, 2.10, 1500),
        )
    )
    clutter = rng.uniform((-1.4, -3.5, 0.2), (1.4, -0.3, 2.3), (300, 3))
    detection = detect_transverse_wall(np.vstack((wall, clutter)), **PARAMS)
    assert detection is not None
    assert abs(detection.distance - 1.03) < 0.03
    assert detection.vertical_span > 1.0
    assert detection.cross_span > 1.0


def test_rejects_horizontal_bed_surface():
    rng = np.random.default_rng(8)
    bed = np.column_stack(
        (
            rng.uniform(-0.5, 0.5, 1500),
            rng.uniform(-2.5, -0.4, 1500),
            rng.normal(0.65, 0.005, 1500),
        )
    )
    assert detect_transverse_wall(bed, **PARAMS) is None


def test_rejects_sparse_vertical_structure():
    rng = np.random.default_rng(9)
    sparse = np.column_stack(
        (
            rng.uniform(-1.0, 1.0, 50),
            rng.normal(-1.0, 0.005, 50),
            rng.uniform(0.2, 2.2, 50),
        )
    )
    assert detect_transverse_wall(sparse, **PARAMS) is None


def test_rejects_high_horizontal_surface_with_sparse_low_outliers():
    rng = np.random.default_rng(10)
    high_surface = np.column_stack(
        (
            rng.uniform(-1.0, 1.0, 400),
            rng.normal(-0.8, 0.008, 400),
            rng.normal(2.04, 0.005, 400),
        )
    )
    low_outliers = np.column_stack(
        (
            rng.uniform(-1.0, 1.0, 50),
            rng.normal(-0.8, 0.008, 50),
            rng.choice((0.25, 0.75, 1.25), 50),
        )
    )
    cloud = np.vstack((high_surface, low_outliers))
    assert detect_transverse_wall(cloud, **PARAMS) is None
