from transfer_robot.docking_console import stop_service_for_phase


def test_stop_keeps_diff_mode_during_retract():
    assert stop_service_for_phase("RETRACT") == "retract"


def test_stop_disables_normal_docking_control():
    assert stop_service_for_phase("PREALIGN") == "enable"
    assert stop_service_for_phase("INSERT_TOP") == "enable"
    assert stop_service_for_phase("UNKNOWN") == "enable"
