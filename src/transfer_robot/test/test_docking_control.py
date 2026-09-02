from math import pi, radians, tan

from transfer_robot.docking_control import (
    compute_diff_angle_command,
    compute_insert_command,
    compute_prealign_command,
    compute_retract_command,
    evaluate_top_handoff_test,
    insertion_detection_phase,
)


PREALIGN = dict(
    target_cross=0.0,
    aligned_cross=0.08,
    max_linear=0.08,
    min_linear=0.02,
    k_cross=0.8,
)

ANGLE_ALIGN = dict(
    target_bed_yaw=-pi / 2.0,
    aligned_yaw=0.087,
    max_angular=0.08,
    min_angular=0.02,
    k_yaw=0.6,
)

INSERT = dict(
    target_entrance_along=-1.02,
    target_cross=0.0,
    stop_along=0.05,
    stop_cross=0.05,
    stop_yaw=0.06,
    max_linear=0.08,
    max_reverse=0.04,
    max_angular=0.08,
    k_along=0.55,
    k_cross=0.7,
    k_yaw=0.6,
    cross_deadband=0.02,
    yaw_deadband=0.02,
    target_bed_yaw=-pi / 2.0,
)


def test_stage_one_ackermann_moves_x_to_bed_board_center():
    cmd = compute_prealign_command(0.10, -1.0, -pi / 2.0, **PREALIGN)
    assert cmd.state == "CENTER_ACKERMANN"
    assert cmd.linear > 0.0
    assert cmd.lateral == 0.0
    assert cmd.angular == 0.0


def test_stage_one_ackermann_does_not_correct_yaw():
    cmd = compute_prealign_command(0.30, -1.0, -pi / 2.0 + radians(8.0), **PREALIGN)
    assert cmd.state == "CENTER_ACKERMANN"
    assert cmd.linear > 0.0
    assert cmd.lateral == 0.0
    assert cmd.angular == 0.0


def test_stage_one_completion_requires_only_cross_centering():
    yaw = -pi / 2.0 + radians(8.0)
    # This non-zero base X lies on the tilted bed's perpendicular centreline.
    entrance_x = tan(radians(8.0))
    cmd = compute_prealign_command(entrance_x, -1.0, yaw, **PREALIGN)
    assert cmd.state == "CENTER_HOLD"
    assert cmd.linear == 0.0
    assert cmd.lateral == 0.0
    assert cmd.angular == 0.0


def test_stage_one_respects_configured_cross_offset():
    params = dict(PREALIGN)
    params["target_cross"] = 0.10
    cmd = compute_prealign_command(0.0, -1.20, -pi / 2.0, **params)
    assert cmd.state == "CENTER_ACKERMANN"
    assert cmd.linear < 0.0
    assert cmd.lateral == 0.0


def test_stage_one_uses_bed_cross_error_when_bed_is_rotated():
    cmd = compute_prealign_command(0.30, -1.0, -pi / 2.0 + radians(8.0), **PREALIGN)
    assert cmd.state == "CENTER_ACKERMANN"
    assert cmd.cross_error > 0.15
    assert cmd.linear > 0.0
    assert cmd.lateral == 0.0


def test_stage_two_diff_rotates_in_place_to_align_yaw():
    cmd = compute_diff_angle_command(
        0.0, -1.0, -pi / 2.0 + radians(8.0), **ANGLE_ALIGN
    )
    assert cmd.state == "ANGLE_ALIGN_DIFF"
    assert cmd.linear == 0.0
    assert cmd.lateral == 0.0
    assert 0.0 < cmd.angular <= ANGLE_ALIGN["max_angular"]


def test_stage_two_diff_holds_when_yaw_aligned():
    cmd = compute_diff_angle_command(0.0, -1.0, -pi / 2.0, **ANGLE_ALIGN)
    assert cmd.state == "ANGLE_HOLD"
    assert cmd.linear == 0.0
    assert cmd.lateral == 0.0
    assert cmd.angular == 0.0


def test_stage_two_inserts_straight():
    cmd = compute_insert_command(0.0, -0.50, -pi / 2.0, **INSERT)
    assert cmd.state == "INSERT"
    assert cmd.linear == 0.0
    assert -0.08 <= cmd.lateral < 0.0
    assert abs(cmd.angular) < 1.0e-9


def test_retract_moves_opposite_to_insert_in_diff_mode():
    cmd = compute_retract_command(0.02)
    assert cmd.state == "RETRACT"
    assert cmd.linear == 0.0
    assert cmd.lateral == 0.02
    assert cmd.angular == 0.0


def test_retract_speed_is_always_opposite_to_insert():
    assert compute_retract_command(-0.02).lateral == 0.02


def test_retract_reverses_cross_correction_while_moving_out():
    cmd = compute_retract_command(
        0.02,
        entrance_x=0.10,
        entrance_y=-0.50,
        bed_yaw=-pi / 2.0,
        cross_deadband=0.01,
    )
    assert cmd.state == "RETRACT_ALIGN"
    assert cmd.lateral > 0.0
    assert cmd.angular < 0.0
    assert cmd.cross_error > 0.0


def test_retract_corrects_yaw_in_the_same_direction_as_angle_alignment():
    cmd = compute_retract_command(
        0.02,
        entrance_x=0.0,
        entrance_y=0.0,
        bed_yaw=-pi / 2.0 + radians(4.0),
        yaw_deadband=0.01,
    )
    assert cmd.angular > 0.0


def test_retract_ignores_small_pose_noise():
    cmd = compute_retract_command(
        0.02,
        entrance_x=0.005,
        entrance_y=0.0,
        bed_yaw=-pi / 2.0 + radians(0.5),
        cross_deadband=0.01,
        yaw_deadband=radians(1.0),
    )
    assert cmd.state == "RETRACT"
    assert cmd.angular == 0.0


def test_stage_two_diff_corrects_while_inserting():
    cmd = compute_insert_command(0.10, -0.50, -pi / 2.0 + radians(8.0), **INSERT)
    assert cmd.linear == 0.0
    assert -0.08 <= cmd.lateral < 0.0
    assert 0.0 < cmd.angular <= 0.08


def test_stage_two_ignores_small_detection_noise_while_inserting():
    cmd = compute_insert_command(
        0.005, -0.50, -pi / 2.0 + radians(0.5), **INSERT
    )
    assert cmd.lateral < 0.0
    assert cmd.angular == 0.0


def test_stage_two_lateral_speed_decreases_near_goal():
    far = compute_insert_command(0.0, -0.50, -pi / 2.0, **INSERT)
    near = compute_insert_command(0.0, 0.90, -pi / 2.0, **INSERT)
    assert 0.0 < abs(near.lateral) < abs(far.lateral)


def test_stage_two_diff_angular_rate_uses_independent_limit():
    cmd = compute_insert_command(0.10, -0.50, -pi / 2.0 + radians(8.0), **INSERT)
    assert abs(cmd.angular) <= INSERT["max_angular"] + 1.0e-9


def test_stage_two_stops_at_bed_center():
    cmd = compute_insert_command(0.0, 1.02, -pi / 2.0, **INSERT)
    assert cmd.state == "DOCKED"
    assert cmd.linear == 0.0
    assert cmd.lateral == 0.0
    assert cmd.angular == 0.0


def test_stage_two_small_overshoot_reverses():
    cmd = compute_insert_command(0.0, 1.12, -pi / 2.0, **INSERT)
    assert cmd.state == "INSERT_CORRECT"
    assert cmd.linear == 0.0
    assert cmd.lateral > 0.0


def test_insert_handoff_test_keeps_running_before_top_or_distance_limit():
    result = evaluate_top_handoff_test(
        "FOOT_RANSAC", current_along=-1.20, insert_start_along=-1.00,
        max_insert_distance=0.35,
    )
    assert result is None


def test_insert_uses_only_foot_before_35_cm():
    assert insertion_detection_phase(-1.349, -1.0, 0.35) == "INSERT_FOOT"


def test_insert_enables_top_at_35_cm():
    assert insertion_detection_phase(-1.35, -1.0, 0.35) == "INSERT_TOP"


def test_insert_handoff_test_stops_when_top_is_acquired():
    result = evaluate_top_handoff_test(
        "TOP_RECTANGLE", current_along=-1.22, insert_start_along=-1.00,
        max_insert_distance=0.35,
    )
    assert result == "TOP_HANDOFF_COMPLETE"


def test_insert_handoff_test_accepts_top_outline_detector():
    result = evaluate_top_handoff_test(
        "TOP_OUTLINES", current_along=-1.22, insert_start_along=-1.00,
        max_insert_distance=0.35,
    )
    assert result == "TOP_HANDOFF_COMPLETE"


def test_insert_handoff_test_stops_at_bounded_distance_without_top():
    result = evaluate_top_handoff_test(
        "FOOT_RANSAC", current_along=-1.36, insert_start_along=-1.00,
        max_insert_distance=0.35,
    )
    assert result == "TOP_HANDOFF_NOT_FOUND"


def test_top_acquisition_wins_at_exact_distance_limit():
    result = evaluate_top_handoff_test(
        "TOP_RECTANGLE", current_along=-1.35, insert_start_along=-1.00,
        max_insert_distance=0.35,
    )
    assert result == "TOP_HANDOFF_COMPLETE"
