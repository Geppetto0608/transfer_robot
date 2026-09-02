from transfer_robot.platform_mode_bridge import MODE_CODES, normalize_motor_status


def test_real_platform_mode_codes():
    assert MODE_CODES["ACKERMANN"] == 1
    assert MODE_CODES["DIFF"] == 2


def test_normalizes_active_motor_modes():
    assert normalize_motor_status("AUTONOMOUS|[ACKERMANN] ready") == "ACKERMANN"
    assert normalize_motor_status("DOCKING|[DIFF] ready") == "DIFF"


def test_normalizes_mode_transitions_and_failure():
    assert normalize_motor_status("ENTERING_DOCKING|[TO_DIFF] moving") == "TO_DIFF"
    assert normalize_motor_status("EXITING_DOCKING|[TO_ACKERMANN] moving") == "TO_ACKERMANN"
    assert normalize_motor_status("FAILED|fault") == "FAILED"
    assert normalize_motor_status("FAILED|[DIFF] Motor feedback stale") == "FAILED"
    assert normalize_motor_status("FAILED|[ACKERMANN] Motor feedback stale") == "FAILED"
