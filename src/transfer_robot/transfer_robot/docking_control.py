from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def wrap_angle(angle: float) -> float:
    return (angle + pi) % (2.0 * pi) - pi


def signed_minimum(value: float, minimum_magnitude: float) -> float:
    """Keep a non-zero command large enough to overcome the platform deadband."""
    if value > 0.0:
        return max(value, minimum_magnitude)
    if value < 0.0:
        return min(value, -minimum_magnitude)
    return 0.0


def insertion_detection_phase(
    current_along: float,
    insert_start_along: float,
    top_handoff_start_distance: float,
) -> str:
    """Select the perception model from distance travelled since INSERT start."""
    travelled = abs(float(current_along) - float(insert_start_along))
    if travelled < max(0.0, float(top_handoff_start_distance)):
        return "INSERT_FOOT"
    return "INSERT_TOP"


def evaluate_top_handoff_test(
    detection_mode: str,
    current_along: float,
    insert_start_along: float,
    max_insert_distance: float,
) -> str | None:
    """Return the bounded FOOT-to-TOP commissioning-test result.

    A confirmed top outline always wins over the travel limit when both
    arrive in the same control cycle. This avoids reporting a failed handoff
    at the exact frame in which the bed top was acquired.
    """
    if detection_mode.strip().upper() in ("TOP_OUTLINES", "TOP_RECTANGLE"):
        return "TOP_HANDOFF_COMPLETE"
    if abs(current_along - insert_start_along) >= max(0.0, max_insert_distance):
        return "TOP_HANDOFF_NOT_FOUND"
    return None


@dataclass(frozen=True)
class DockCommand:
    linear: float
    lateral: float
    angular: float
    state: str
    along_error: float
    cross_error: float


def compute_retract_command(
    speed: float,
    entrance_x: float = 0.0,
    entrance_y: float = 0.0,
    bed_yaw: float = -pi / 2.0,
    target_cross: float = 0.0,
    max_angular: float = 0.05,
    k_cross: float = 0.7,
    k_yaw: float = 0.6,
    cross_deadband: float = 0.02,
    yaw_deadband: float = 0.02,
    target_bed_yaw: float = -pi / 2.0,
) -> DockCommand:
    """Retract in DIFF while holding the detected bed centreline and yaw.

    Retraction reverses the insertion translation, so its cross-track term
    must also reverse.  The yaw term keeps the same sign because positive
    robot yaw always makes the stationary bed yaw decrease in ``base_link``.
    """
    yaw_error = wrap_angle(bed_yaw - target_bed_yaw)
    axis_x, axis_y = cos(bed_yaw), sin(bed_yaw)
    normal_x, normal_y = -axis_y, axis_x
    cross_to_entrance = normal_x * entrance_x + normal_y * entrance_y
    cross_error = cross_to_entrance - target_cross
    cross_control = (
        0.0 if abs(cross_error) <= max(0.0, cross_deadband) else cross_error
    )
    yaw_control = (
        0.0 if abs(yaw_error) <= max(0.0, yaw_deadband) else yaw_error
    )
    angular = clamp(
        -k_cross * cross_control + k_yaw * yaw_control,
        -abs(max_angular),
        abs(max_angular),
    )
    error_scale = clamp(
        1.0 - 1.8 * abs(cross_error) - 1.5 * abs(yaw_error), 0.25, 1.0
    )
    state = "RETRACT_ALIGN" if abs(angular) > 1.0e-9 else "RETRACT"
    return DockCommand(
        0.0,
        abs(float(speed)) * error_scale,
        angular,
        state,
        yaw_error,
        cross_error,
    )


def compute_prealign_command(
    entrance_x: float,
    entrance_y: float,
    bed_yaw: float,
    target_cross: float,
    aligned_cross: float,
    max_linear: float,
    min_linear: float,
    k_cross: float,
) -> DockCommand:
    """Stage 1: centre on the foot-board normal using straight Ackermann motion.

    Heading is deliberately not corrected here. It is aligned in place after
    switching to DIFF, so Ackermann never oscillates between forward/reverse
    steering targets while trying to manufacture a rotation near X == 0.
    """
    bed_axis_x, bed_axis_y = cos(bed_yaw), sin(bed_yaw)
    bed_normal_x, bed_normal_y = -bed_axis_y, bed_axis_x
    along_to_entrance = bed_axis_x * entrance_x + bed_axis_y * entrance_y
    cross_to_entrance = bed_normal_x * entrance_x + bed_normal_y * entrance_y
    along_error = along_to_entrance
    cross_error = cross_to_entrance - target_cross

    if abs(cross_error) <= aligned_cross:
        return DockCommand(
            0.0, 0.0, 0.0, "CENTER_HOLD", along_error, cross_error
        )

    # cross_dot = -linear.x * bed_normal_x for a stationary bed. The detected
    # foot-board is constrained near the target -Y heading, so this projection
    # gives the correct base-X direction without requesting any steering.
    linear = clamp(
        k_cross * cross_error * bed_normal_x, -max_linear, max_linear
    )
    linear = signed_minimum(linear, min_linear)
    return DockCommand(
        linear, 0.0, 0.0, "CENTER_ACKERMANN", along_error, cross_error
    )


def compute_diff_angle_command(
    entrance_x: float,
    entrance_y: float,
    bed_yaw: float,
    target_bed_yaw: float,
    aligned_yaw: float,
    max_angular: float,
    min_angular: float,
    k_yaw: float,
) -> DockCommand:
    """Stage 2: rotate in place in DIFF until the bed yaw is aligned."""
    yaw_error = wrap_angle(bed_yaw - target_bed_yaw)
    if abs(yaw_error) <= aligned_yaw:
        return DockCommand(
            0.0, 0.0, 0.0, "ANGLE_HOLD", yaw_error, entrance_x
        )

    # A positive robot yaw rotates a stationary bed negatively in base_link,
    # so command the same sign as the measured bed-yaw error.
    angular = clamp(k_yaw * yaw_error, -max_angular, max_angular)
    angular = signed_minimum(angular, min_angular)
    angular = clamp(angular, -max_angular, max_angular)
    return DockCommand(
        0.0, 0.0, angular, "ANGLE_ALIGN_DIFF", yaw_error, entrance_x
    )


def compute_insert_command(
    entrance_x: float,
    entrance_y: float,
    bed_yaw: float,
    target_entrance_along: float,
    target_cross: float,
    stop_along: float,
    stop_cross: float,
    stop_yaw: float,
    max_linear: float,
    max_reverse: float,
    max_angular: float,
    k_along: float,
    k_cross: float,
    k_yaw: float,
    cross_deadband: float,
    yaw_deadband: float,
    target_bed_yaw: float = -pi / 2.0,
) -> DockCommand:
    """Stage 2: insert laterally in DIFF mode while correcting heading."""
    yaw_error = wrap_angle(bed_yaw - target_bed_yaw)
    axis_x, axis_y = cos(bed_yaw), sin(bed_yaw)
    normal_x, normal_y = -axis_y, axis_x
    along_to_entrance = axis_x * entrance_x + axis_y * entrance_y
    cross_to_entrance = normal_x * entrance_x + normal_y * entrance_y
    # At the final dock pose the detected entrance normally lies behind
    # base_link, hence target_entrance_along is commonly negative (for example
    # -1.02 m as an initial estimate). Both offsets must be measured on hardware.
    along_error = along_to_entrance - target_entrance_along
    cross_error = cross_to_entrance - target_cross

    if (
        abs(along_error) <= stop_along
        and abs(cross_error) <= stop_cross
        and abs(yaw_error) <= stop_yaw
    ):
        return DockCommand(
            0.0, 0.0, 0.0, "DOCKED", along_error, cross_error
        )

    # DIFF docking fixes both steering axes near +/-90 degrees.  It therefore
    # inserts along base Y and may use angular.z for simultaneous alignment.
    along_speed = clamp(k_along * along_error, -max_reverse, max_linear)
    error_scale = clamp(
        1.0 - 1.8 * abs(cross_error) - 1.5 * abs(yaw_error), 0.25, 1.0
    )
    final_scale = clamp(abs(along_error) / 0.35, 0.18, 1.0)
    along_speed *= min(error_scale, final_scale)
    lateral = along_speed * sin(target_bed_yaw)
    # Do not chase sub-centimetre/degree-scale LiDAR fit noise. Outside the
    # deadband retain the full error so meaningful corrections are not weak.
    cross_control = (
        0.0 if abs(cross_error) <= max(0.0, cross_deadband) else cross_error
    )
    yaw_control = (
        0.0 if abs(yaw_error) <= max(0.0, yaw_deadband) else yaw_error
    )
    angular = clamp(
        k_cross * cross_control + k_yaw * yaw_control,
        -max_angular,
        max_angular,
    )
    if along_speed < 0.0:
        state = "INSERT_CORRECT"
    else:
        state = "FINAL_INSERT" if abs(along_error) < 0.35 else "INSERT"
    return DockCommand(
        0.0, lateral, angular, state, along_error, cross_error
    )
