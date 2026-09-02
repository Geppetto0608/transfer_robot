import numpy as np

from transfer_robot.geometry import (
    BedDetection,
    apply_bed_pose_calibration,
    detect_bed_foot_line,
    detect_bed_side_lines,
    detect_bed_top_outlines,
    detect_bed_top_rectangle,
    detect_bed_pair,
    detect_bed_rectangle,
    euclidean_clusters,
    planar_downsample,
    voxel_downsample,
)


def test_detect_axis_aligned_bed():
    centers = np.array(
        [
            [1.02, 0.34],
            [1.02, -0.34],
            [2.58, 0.34],
            [2.58, -0.34],
            [0.6, 1.2],  # unrelated obstacle
        ],
        dtype=float,
    )
    detection = detect_bed_rectangle(centers)
    assert detection is not None
    assert np.allclose(detection.entrance_center, [1.02, 0.0], atol=1.0e-6)
    assert abs(detection.width - 0.68) < 1.0e-6
    assert abs(detection.length - 1.56) < 1.0e-6
    assert abs(detection.yaw) < 1.0e-6


def test_reject_wrong_width():
    centers = np.array([[1.0, -0.1], [1.0, 0.1], [2.56, -0.1], [2.56, 0.1]])
    assert detect_bed_rectangle(centers) is None


def test_voxel_downsample():
    points = np.array([[0.001, 0.0, 0.0], [0.009, 0.0, 0.0], [0.1, 0.0, 0.0]])
    result = voxel_downsample(points, 0.02)
    assert len(result) == 2


def test_planar_downsample_collapses_vertical_projection_density():
    points = np.array(
        [[0.001, 0.001], [0.009, 0.008], [0.041, 0.0]], dtype=float
    )
    result = planar_downsample(points, 0.02)
    assert result.shape == (2, 2)


def test_detect_partial_rotated_bed_top_rectangle_from_merged_occupancy():
    rng = np.random.default_rng(211)
    entrance = np.array([0.18, -1.55])
    yaw = -np.pi / 2.0 + 0.07
    axis = np.array([np.cos(yaw), np.sin(yaw)])
    normal = np.array([-axis[1], axis[0]])
    # Only a middle portion of the filled top is visible. Repeated noisy
    # samples emulate overlapping returns from the three merged LiDARs.
    along = np.linspace(0.10, 1.35, 55)
    cross = np.linspace(-0.45, 0.45, 31)
    aa, cc = np.meshgrid(along, cross)
    top = entrance + aa.reshape(-1, 1) * axis + cc.reshape(-1, 1) * normal
    merged = np.vstack(
        [top + rng.normal(0.0, 0.006, top.shape) for _ in range(3)]
    )
    clutter = np.column_stack(
        (rng.uniform(-1.0, 1.0, 300), rng.uniform(0.2, 1.2, 300))
    )

    detection = detect_bed_top_rectangle(
        np.vstack((merged, clutter)),
        reference_entrance=entrance + np.array([0.02, -0.01]),
        reference_yaw=yaw + 0.04,
        expected_width=0.90,
        width_tolerance=0.16,
        expected_length=2.04,
        min_observed_length=0.55,
        min_points=40,
        max_yaw_delta=0.20,
        projection_leaf=0.03,
    )

    assert detection is not None
    assert np.linalg.norm(detection.entrance_center - entrance) < 0.18
    assert abs(detection.width - 0.90) < 0.10
    assert abs(((detection.yaw - yaw + np.pi) % (2.0 * np.pi)) - np.pi) < 0.08
    assert detection.confidence > 0.45
    assert len(detection.evidence_points) >= 40


def test_reject_top_patch_without_expected_bed_width():
    along = np.linspace(0.0, 1.1, 40)
    cross = np.linspace(-0.12, 0.12, 12)
    aa, cc = np.meshgrid(along, cross)
    points = np.column_stack((aa.ravel(), cc.ravel()))
    detection = detect_bed_top_rectangle(
        points,
        reference_entrance=np.zeros(2),
        reference_yaw=0.0,
        expected_width=0.90,
        width_tolerance=0.20,
        min_points=30,
    )
    assert detection is None


def test_detect_top_outlines_from_two_rotated_edges():
    rng = np.random.default_rng(212)
    entrance = np.array([0.18, -1.55])
    yaw = -np.pi / 2.0 + 0.07
    axis = np.array([np.cos(yaw), np.sin(yaw)])
    normal = np.array([-axis[1], axis[0]])
    along = np.linspace(0.02, 1.85, 150)
    edges = np.vstack(
        (
            entrance + along[:, None] * axis + 0.45 * normal,
            entrance + along[:, None] * axis - 0.45 * normal,
        )
    )
    edges += rng.normal(0.0, 0.006, edges.shape)
    detection = detect_bed_top_outlines(
        edges,
        reference_entrance=entrance + np.array([0.02, -0.01]),
        reference_yaw=yaw + 0.04,
        expected_width=0.90,
        width_tolerance=0.16,
        bed_length=2.04,
        min_points=20,
        min_observed_length=0.55,
        min_abs_direction=0.82,
    )
    assert detection is not None
    assert abs(((detection.yaw - yaw + np.pi) % (2.0 * np.pi)) - np.pi) < 0.06
    assert abs(detection.width - 0.90) < 0.08
    assert len(detection.evidence_points) >= 20


def test_reacquire_top_outlines_without_a_previous_foot_pose():
    rng = np.random.default_rng(213)
    entrance = np.array([0.10, 0.65])
    yaw = -np.pi / 2.0 + 0.04
    axis = np.array([np.cos(yaw), np.sin(yaw)])
    normal = np.array([-axis[1], axis[0]])
    along = np.linspace(0.10, 1.75, 140)
    edges = np.vstack(
        (
            entrance + along[:, None] * axis + 0.37 * normal,
            entrance + along[:, None] * axis - 0.37 * normal,
        )
    )
    edges += rng.normal(0.0, 0.006, edges.shape)
    detection = detect_bed_top_outlines(
        edges,
        reference_entrance=np.zeros(2),
        reference_yaw=-np.pi / 2.0,
        expected_width=0.74,
        width_tolerance=0.12,
        bed_length=2.04,
        min_points=20,
        min_observed_length=0.55,
        min_abs_direction=0.88,
        tracking_position_weight=0.0,
    )
    assert detection is not None
    assert abs(detection.width - 0.74) < 0.06
    yaw_error = (detection.yaw - yaw + np.pi) % (2.0 * np.pi) - np.pi
    assert abs(yaw_error) < 0.06


def test_apply_bed_pose_calibration_aligns_top_model_to_foot_reference():
    raw = BedDetection(
        entrance_center=np.array([0.004, -0.167]),
        far_center=np.array([-0.121, -2.203]),
        yaw=np.deg2rad(-93.5),
        confidence=0.62,
        corners=np.zeros((4, 2)),
        width=0.74,
        length=2.04,
        evidence_points=np.array([[0.3, -0.2], [-0.4, -0.2]]),
    )
    reference = np.array([-0.021, -0.695])
    reference_yaw = np.deg2rad(-90.33)
    axis = np.array([np.cos(raw.yaw), np.sin(raw.yaw)])
    normal = np.array([-axis[1], axis[0]])
    delta = reference - raw.entrance_center
    calibrated = apply_bed_pose_calibration(
        raw,
        float(delta @ axis),
        float(delta @ normal),
        reference_yaw - raw.yaw,
    )
    assert np.allclose(calibrated.entrance_center, reference, atol=1.0e-8)
    assert abs(calibrated.yaw - reference_yaw) < 1.0e-8
    assert np.allclose(calibrated.evidence_points, raw.evidence_points)


def test_detect_visible_entrance_pair():
    centers = np.array([[0.98, -0.337], [0.98, 0.337], [0.35, -0.51]])
    detection = detect_bed_pair(centers)
    assert detection is not None
    assert np.allclose(detection.entrance_center, [0.98, 0.0], atol=1.0e-6)
    assert abs(detection.width - 0.674) < 1.0e-6
    assert abs(detection.yaw) < 1.0e-6


def test_pair_tracking_does_not_switch_to_rear_legs():
    centers = np.array([[-0.70, -0.34], [-0.70, 0.34], [0.86, -0.34], [0.86, 0.34]])
    detection = detect_bed_pair(centers, tracking_reference=np.array([-0.68, 0.0]))
    assert detection is not None
    assert np.allclose(detection.entrance_center, [-0.70, 0.0], atol=1.0e-6)


def test_detect_bed_foot_frame_line_without_leg_candidates():
    rng = np.random.default_rng(7)
    y = np.linspace(-0.475, 0.475, 80)
    foot = np.column_stack((1.0 + rng.normal(0.0, 0.008, len(y)), y))
    unrelated = np.column_stack((np.linspace(1.2, 2.0, 40), np.full(40, 1.1)))
    points = np.vstack((foot, unrelated))
    clusters = euclidean_clusters(points, tolerance=0.08, min_points=8, max_points=1000)
    detection = detect_bed_foot_line(points, clusters)
    assert detection is not None
    assert np.allclose(detection.entrance_center, [1.0, 0.0], atol=0.04)
    assert abs(detection.width - 0.855) < 0.08
    assert abs(detection.yaw) < 0.05
    assert len(detection.evidence_points) >= 20
    assert np.max(np.abs(detection.evidence_points[:, 0] - 1.0)) < 0.06


def test_detect_x_axis_bed_foot_frame_for_bag_replay():
    rng = np.random.default_rng(11)
    x = np.linspace(-0.44, 0.44, 90)
    foot = np.column_stack((x, 0.45 + rng.normal(0.0, 0.006, len(x))))
    unrelated = np.column_stack((np.full(40, 0.9), np.linspace(0.7, 1.3, 40)))
    points = np.vstack((foot, unrelated))
    clusters = euclidean_clusters(points, tolerance=0.08, min_points=8, max_points=1000)
    detection = detect_bed_foot_line(
        points,
        clusters,
        expected_width=0.88,
        width_tolerance=0.15,
        line_axis="X",
        min_abs_direction=0.88,
    )
    assert detection is not None
    assert np.allclose(detection.entrance_center, [0.0, 0.45], atol=0.04)
    assert abs(detection.width - 0.792) < 0.08
    assert abs(detection.yaw - np.pi / 2.0) < 0.05
    assert len(detection.evidence_points) >= 20


def test_detect_x_axis_bed_foot_frame_toward_robot_negative_y():
    rng = np.random.default_rng(17)
    x = np.linspace(-0.44, 0.44, 90)
    foot = np.column_stack((x, -0.65 + rng.normal(0.0, 0.006, len(x))))
    clusters = euclidean_clusters(foot, tolerance=0.08, min_points=8, max_points=1000)
    detection = detect_bed_foot_line(
        foot,
        clusters,
        expected_width=0.88,
        width_tolerance=0.15,
        line_axis="X",
        min_abs_direction=0.88,
        bed_axis_sign=-1.0,
    )
    assert detection is not None
    assert abs(detection.yaw + np.pi / 2.0) < 0.05
    assert detection.far_center[1] < detection.entrance_center[1]


def test_tracked_foot_does_not_jump_to_nearer_parallel_structure():
    rng = np.random.default_rng(23)
    x = np.linspace(-0.44, 0.44, 100)
    tracked = np.column_stack(
        (x, -1.22 + rng.normal(0.0, 0.006, len(x)))
    )
    nearer = np.column_stack(
        (x, -1.02 + rng.normal(0.0, 0.006, len(x)))
    )
    points = np.vstack((tracked, nearer))
    clusters = euclidean_clusters(
        points, tolerance=0.08, min_points=8, max_points=1000
    )
    detection = detect_bed_foot_line(
        points,
        clusters,
        expected_width=0.88,
        width_tolerance=0.15,
        line_axis="X",
        min_abs_direction=0.88,
        bed_axis_sign=-1.0,
        tracking_reference=np.array([0.0, -1.22]),
        tracking_yaw=-np.pi / 2.0,
        tracking_position_weight=4.0,
        tracking_yaw_weight=2.5,
    )
    assert detection is not None
    assert abs(detection.entrance_center[1] + 1.22) < 0.04


def test_detect_bed_upper_side_lines_during_insert():
    rng = np.random.default_rng(19)
    x = np.linspace(1.0, 3.04, 120)
    left = np.column_stack((x, 0.45 + rng.normal(0.0, 0.006, len(x))))
    right = np.column_stack((x, -0.45 + rng.normal(0.0, 0.006, len(x))))
    unrelated = np.column_stack((np.full(35, 1.7), np.linspace(0.8, 1.3, 35)))
    points = np.vstack((left, right, unrelated))
    clusters = euclidean_clusters(points, tolerance=0.08, min_points=8, max_points=1000)
    detection = detect_bed_side_lines(points, clusters)
    assert detection is not None
    assert np.allclose(detection.entrance_center, [1.0, 0.0], atol=0.08)
    assert abs(detection.width - 0.90) < 0.04
    assert abs(detection.yaw) < 0.03
    assert len(detection.evidence_points) >= 12
    assert np.ptp(detection.evidence_points[:, 1]) > 0.80


def test_detect_thick_upper_and_lower_bed_structure_during_insert():
    rng = np.random.default_rng(101)
    x = np.linspace(0.70, 2.74, 150)
    layers = []

    # Multiple upper/lower returns form thick side bands. Repeating them
    # emulates several Z layers projected into XY by the planar detector.
    for side in (-0.45, 0.45):
        inward = 1.0 if side < 0.0 else -1.0
        for offset in np.linspace(0.0, 0.12, 7):
            for _ in range(3):
                layers.append(
                    np.column_stack(
                        (
                            x + rng.normal(0.0, 0.004, len(x)),
                            np.full(len(x), side + inward * offset)
                            + rng.normal(0.0, 0.005, len(x)),
                        )
                    )
                )

    # Lower cross-members connect both sides and make one filled-looking
    # cluster. The detector must still choose the symmetric outer envelope.
    for center_x in np.linspace(0.85, 2.55, 6):
        y = np.linspace(-0.44, 0.44, 90)
        layers.append(
            np.column_stack(
                (
                    np.full(len(y), center_x) + rng.normal(0.0, 0.008, len(y)),
                    y,
                )
            )
        )

    points = planar_downsample(np.vstack(layers), 0.025)
    clusters = euclidean_clusters(
        points, tolerance=0.10, min_points=8, max_points=10000
    )
    detection = detect_bed_side_lines(
        points,
        clusters,
        expected_width=0.90,
        width_tolerance=0.20,
        min_points=30,
        min_observed_length=0.55,
        min_abs_direction_x=0.88,
        bin_min_points=4,
        boundary_min_bins=5,
        symmetry_weight=2.0,
        envelope_preference=0.35,
        tracking_position_weight=1.2,
        tracking_reference=np.array([0.70, 0.0]),
        tracking_yaw=0.0,
    )

    assert detection is not None
    assert abs(detection.entrance_center[1]) < 0.04
    assert abs(detection.width - 0.90) < 0.08
    assert abs(detection.yaw) < 0.04
    assert len(detection.evidence_points) >= 10
