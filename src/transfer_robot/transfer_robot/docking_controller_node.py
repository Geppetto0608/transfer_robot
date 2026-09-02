from __future__ import annotations

from math import atan2, cos, sin
from time import monotonic

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32, String
from std_srvs.srv import SetBool

from .docking_control import (
    compute_diff_angle_command,
    compute_insert_command,
    compute_prealign_command,
    compute_retract_command,
    evaluate_top_handoff_test,
    insertion_detection_phase,
)


class DockingControllerNode(Node):
    # Actual platform semantics:
    #   DIFF      = wheels near +/-90 deg, lateral Y motion + in-place yaw
    #   ACKERMANN = 4WS X motion with bicycle-like curvature
    # Docking sequence: ACKERMANN X centering -> stop -> DIFF yaw alignment
    # -> DIFF Y insertion.
    MODE_DIFF = "DIFF"
    MODE_ACKERMANN = "ACKERMANN"

    def __init__(self) -> None:
        super().__init__("docking_controller")
        self.declare_parameter("pose_topic", "/detected_dock_pose")
        self.declare_parameter("confidence_topic", "/bed_detection/confidence")
        self.declare_parameter("detection_mode_topic", "/bed_detection/mode")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("mode_feedback_topic", "/drive_mode_feedback")
        self.declare_parameter("pose_frame", "base_link")
        self.declare_parameter("require_mode_feedback", False)
        self.declare_parameter("enabled", False)
        # Commissioning guard: ACKERMANN centering + DIFF yaw alignment, then
        # hold without insertion. Disable only for a complete docking run.
        self.declare_parameter("prealign_only", True)
        # First insertion commissioning test: stop as soon as FOOT -> TOP
        # succeeds, or after a bounded travel if TOP is never acquired.
        self.declare_parameter("stop_on_top_handoff", True)
        self.declare_parameter("top_handoff_start_distance", 0.35)
        self.declare_parameter("top_handoff_test_max_insert_distance", 0.35)
        self.declare_parameter("min_confidence", 0.45)
        self.declare_parameter("detection_timeout", 0.6)
        self.declare_parameter("prealign_cross_tolerance", 0.08)
        self.declare_parameter("prealign_hold_frames", 3)
        self.declare_parameter("mode_switch_stop_frames", 5)
        self.declare_parameter("mode_switch_settle_frames", 5)
        self.declare_parameter("mode_switch_timeout", 8.0)
        self.declare_parameter("prealign_max_linear", 0.08)
        self.declare_parameter("prealign_min_linear", 0.02)
        self.declare_parameter("prealign_k_cross", 0.8)
        self.declare_parameter("angle_align_yaw_tolerance", 0.087)
        self.declare_parameter("angle_align_hold_frames", 3)
        self.declare_parameter("angle_align_max_angular", 0.18)
        self.declare_parameter("angle_align_min_angular", 0.04)
        self.declare_parameter("angle_align_k_yaw", 0.6)
        self.declare_parameter("target_bed_yaw", -1.5707963267948966)
        self.declare_parameter("target_entrance_along", -1.02)
        self.declare_parameter("target_cross", 0.0)
        self.declare_parameter("stop_along", 0.05)
        self.declare_parameter("stop_cross", 0.05)
        self.declare_parameter("stop_yaw", 0.06)
        self.declare_parameter("insert_max_speed", 0.08)
        self.declare_parameter("insert_max_reverse_speed", 0.04)
        self.declare_parameter("insert_max_angular", 0.08)
        self.declare_parameter("insert_k_along", 0.55)
        self.declare_parameter("insert_k_cross", 0.7)
        self.declare_parameter("insert_k_yaw", 0.6)
        self.declare_parameter("insert_cross_deadband", 0.02)
        self.declare_parameter("insert_yaw_deadband", 0.02)
        self.declare_parameter("insert_abort_cross", 0.25)
        self.declare_parameter("insert_abort_yaw", 0.20)
        self.declare_parameter("retract_speed", 0.02)
        self.declare_parameter("retract_max_angular", 0.05)
        self.declare_parameter("retract_k_cross", 0.7)
        self.declare_parameter("retract_k_yaw", 0.6)
        self.declare_parameter("retract_cross_deadband", 0.02)
        self.declare_parameter("retract_yaw_deadband", 0.02)
        self.declare_parameter("retract_abort_cross", 0.25)
        self.declare_parameter("retract_abort_yaw", 0.20)
        self.declare_parameter("control_rate", 20.0)

        self.enabled = bool(self.get_parameter("enabled").value)
        self.retracting = False
        self.retract_hold = False
        self.pose: PoseStamped | None = None
        self.confidence = 0.0
        self.last_pose_time = 0.0
        self.phase = "PRE_ALIGN"
        self.align_hold_count = 0
        self.angle_hold_count = 0
        self.mode_switch_count = 0
        self.last_state = "DISABLED" if not self.enabled else "WAITING"
        self.mode_feedback = "UNKNOWN"
        self.detection_mode_feedback = "UNKNOWN"
        self.insert_start_along: float | None = None
        self.top_handoff_search_start_along: float | None = None
        self.mode_switch_started = 0.0

        self.create_subscription(
            PoseStamped, str(self.get_parameter("pose_topic").value), self.pose_callback, 10
        )
        self.create_subscription(
            Float32, str(self.get_parameter("confidence_topic").value), self.confidence_callback, 10
        )
        self.create_subscription(
            String,
            str(self.get_parameter("detection_mode_topic").value),
            self.detection_mode_callback,
            10,
        )
        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("mode_feedback_topic").value),
            self.mode_feedback_callback,
            latched_qos,
        )
        self.cmd_pub = self.create_publisher(Twist, str(self.get_parameter("cmd_vel_topic").value), 10)
        self.status_pub = self.create_publisher(String, "/docking_controller/status", 10)
        self.mode_pub = self.create_publisher(String, "/drive_mode", 10)
        self.phase_pub = self.create_publisher(String, "/docking_phase", 10)
        self.enable_service = self.create_service(SetBool, "/docking_controller/enable", self.set_enabled)
        self.retract_service = self.create_service(
            SetBool, "/docking_controller/retract", self.set_retracting
        )
        rate = max(1.0, float(self.get_parameter("control_rate").value))
        self.timer = self.create_timer(1.0 / rate, self.control_loop)
        self.get_logger().info("Docking controller ready (disabled for safety)")

    def p(self, name: str):
        return self.get_parameter(name).value

    def pose_callback(self, message: PoseStamped) -> None:
        expected_frame = str(self.get_parameter("pose_frame").value)
        if message.header.frame_id and message.header.frame_id != expected_frame:
            self.get_logger().warning(
                f"Ignoring dock pose in frame '{message.header.frame_id}', "
                f"expected '{expected_frame}'",
                throttle_duration_sec=2.0,
            )
            return
        self.pose = message
        self.last_pose_time = monotonic()

    def confidence_callback(self, message: Float32) -> None:
        self.confidence = float(message.data)

    def mode_feedback_callback(self, message: String) -> None:
        self.mode_feedback = message.data.strip().upper()

    def detection_mode_callback(self, message: String) -> None:
        self.detection_mode_feedback = message.data.strip().upper()

    def publish_stop(self, state: str) -> None:
        self.cmd_pub.publish(Twist())
        self.publish_state(state)

    def publish_mode(self, mode: str) -> None:
        self.mode_pub.publish(String(data=mode))

    def publish_phase(self, phase: str) -> None:
        self.phase_pub.publish(String(data=phase))

    def publish_state(self, state: str) -> None:
        self.status_pub.publish(String(data=state))
        if state != self.last_state:
            self.get_logger().info(f"Docking state: {state}")
            self.last_state = state

    def set_enabled(self, request: SetBool.Request, response: SetBool.Response) -> SetBool.Response:
        self.retracting = False
        self.retract_hold = False
        self.enabled = bool(request.data)
        self.pose = None
        self.last_pose_time = 0.0
        self.phase = "PRE_ALIGN"
        self.align_hold_count = 0
        self.angle_hold_count = 0
        self.mode_switch_count = 0
        self.mode_switch_started = 0.0
        self.detection_mode_feedback = "UNKNOWN"
        self.insert_start_along = None
        self.top_handoff_search_start_along = None
        if not self.enabled:
            self.publish_mode(self.MODE_ACKERMANN)
            self.publish_phase("PREALIGN")
            self.publish_stop("DISABLED")
        else:
            self.publish_mode(self.MODE_ACKERMANN)
            self.publish_phase("PREALIGN")
            self.publish_stop("WAITING")
        response.success = True
        response.message = "docking controller enabled" if self.enabled else "docking controller disabled"
        return response

    def set_retracting(
        self, request: SetBool.Request, response: SetBool.Response
    ) -> SetBool.Response:
        self.enabled = False
        self.retracting = bool(request.data)
        # A paused retract keeps DIFF and the current perception phase so it
        # can resume without centring the steering or losing the TOP tracker.
        self.retract_hold = True
        self.publish_mode(self.MODE_DIFF)
        self.publish_phase("RETRACT")
        self.publish_stop("RETRACT_WAITING_DIFF" if self.retracting else "RETRACT_STOPPED")
        response.success = True
        response.message = (
            "retract mode enabled" if self.retracting else "retract mode stopped"
        )
        return response

    def control_loop(self) -> None:
        if self.retracting:
            self.publish_mode(self.MODE_DIFF)
            self.publish_phase("RETRACT")
            if bool(self.p("require_mode_feedback")) and self.mode_feedback == "FAILED":
                self.publish_stop("MOTOR_MODE_FAILED")
                return
            if bool(self.p("require_mode_feedback")) and self.mode_feedback != self.MODE_DIFF:
                self.publish_stop("RETRACT_WAITING_DIFF")
                return
            if self.pose is None:
                self.publish_stop("RETRACT_WAITING_DETECTION")
                return
            if monotonic() - self.last_pose_time > float(self.p("detection_timeout")):
                self.publish_stop("LOST")
                return
            if self.confidence < float(self.p("min_confidence")):
                self.publish_stop("LOW_CONFIDENCE")
                return
            q = self.pose.pose.orientation
            yaw = atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z),
            )
            command = compute_retract_command(
                speed=float(self.p("retract_speed")),
                entrance_x=float(self.pose.pose.position.x),
                entrance_y=float(self.pose.pose.position.y),
                bed_yaw=yaw,
                target_cross=float(self.p("target_cross")),
                max_angular=float(self.p("retract_max_angular")),
                k_cross=float(self.p("retract_k_cross")),
                k_yaw=float(self.p("retract_k_yaw")),
                cross_deadband=float(self.p("retract_cross_deadband")),
                yaw_deadband=float(self.p("retract_yaw_deadband")),
                target_bed_yaw=float(self.p("target_bed_yaw")),
            )
            if (
                abs(command.cross_error) > float(self.p("retract_abort_cross"))
                or abs(command.along_error) > float(self.p("retract_abort_yaw"))
            ):
                self.publish_stop("RETRACT_ALIGNMENT_LOST")
                return
            twist = Twist()
            twist.linear.y = command.lateral
            twist.angular.z = command.angular
            self.cmd_pub.publish(twist)
            self.publish_state(command.state)
            return
        if not self.enabled:
            if self.retract_hold:
                self.publish_mode(self.MODE_DIFF)
                self.publish_phase("RETRACT")
                self.publish_stop("RETRACT_STOPPED")
                return
            # Keep the disabled platform in the known prealignment mode. The
            # bridge de-duplicates this continuous request before motor_node.
            self.publish_mode(self.MODE_ACKERMANN)
            self.publish_phase("PREALIGN")
            self.publish_stop("DISABLED")
            return

        # These are latched software-stop states.  Keep their result visible
        # even if perception disappears after the robot has already stopped.
        if self.phase in ("TOP_HANDOFF_TEST_COMPLETE", "TOP_HANDOFF_TEST_LIMIT"):
            self.publish_mode(self.MODE_DIFF)
            self.publish_phase("INSERT_TOP")
            self.publish_stop(
                "TOP_HANDOFF_COMPLETE"
                if self.phase == "TOP_HANDOFF_TEST_COMPLETE"
                else "TOP_HANDOFF_NOT_FOUND"
            )
            return

        if self.pose is None:
            self.publish_phase("PREALIGN" if self.phase != "DOCKED" else "DOCKED")
            self.publish_stop("WAITING")
            return
        if monotonic() - self.last_pose_time > float(self.p("detection_timeout")):
            self.publish_stop("LOST")
            return
        if self.confidence < float(self.p("min_confidence")):
            self.publish_stop("LOW_CONFIDENCE")
            return
        if bool(self.p("require_mode_feedback")) and self.mode_feedback == "FAILED":
            self.publish_stop("MOTOR_MODE_FAILED")
            return

        if self.phase == "DOCKED":
            self.publish_mode(self.MODE_DIFF)
            self.publish_phase("DOCKED")
            self.publish_stop("DOCKED")
            return

        if self.phase == "MODE_SWITCH_STOP":
            self.publish_mode(self.MODE_ACKERMANN)
            self.publish_phase("PREALIGN")
            self.publish_stop("MODE_SWITCH_STOP")
            self.mode_switch_count += 1
            if self.mode_switch_count >= int(self.p("mode_switch_stop_frames")):
                self.phase = "MODE_SWITCH_SETTLE"
                self.mode_switch_count = 0
                self.mode_switch_started = monotonic()
            return

        if self.phase == "MODE_SWITCH_SETTLE":
            self.publish_mode(self.MODE_DIFF)
            self.publish_phase("PREALIGN")
            if bool(self.p("require_mode_feedback")):
                if self.mode_feedback == self.MODE_DIFF:
                    self.mode_switch_count += 1
                else:
                    self.mode_switch_count = 0
                    if monotonic() - self.mode_switch_started > float(self.p("mode_switch_timeout")):
                        self.publish_stop("MODE_SWITCH_TIMEOUT")
                    else:
                        self.publish_stop("MODE_SWITCH_SETTLE")
                    return
            else:
                self.mode_switch_count += 1
            self.publish_stop("MODE_SWITCH_SETTLE")
            if self.mode_switch_count >= int(self.p("mode_switch_settle_frames")):
                self.phase = "ANGLE_ALIGN"
                self.mode_switch_count = 0
            return

        q = self.pose.pose.orientation
        yaw = atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        entrance_x = float(self.pose.pose.position.x)
        entrance_y = float(self.pose.pose.position.y)
        if self.phase == "PRE_ALIGN":
            self.publish_mode(self.MODE_ACKERMANN)
            self.publish_phase("PREALIGN")
            if bool(self.p("require_mode_feedback")) and self.mode_feedback != self.MODE_ACKERMANN:
                self.publish_stop("WAITING_ACKERMANN_MODE")
                return
            command = compute_prealign_command(
                entrance_x=entrance_x,
                entrance_y=entrance_y,
                bed_yaw=yaw,
                target_cross=float(self.p("target_cross")),
                aligned_cross=float(self.p("prealign_cross_tolerance")),
                max_linear=float(self.p("prealign_max_linear")),
                min_linear=float(self.p("prealign_min_linear")),
                k_cross=float(self.p("prealign_k_cross")),
            )
            if command.state == "CENTER_HOLD":
                self.align_hold_count += 1
                if self.align_hold_count >= int(self.p("prealign_hold_frames")):
                    self.phase = "MODE_SWITCH_STOP"
                    self.align_hold_count = 0
                    self.publish_stop("CENTERED")
                    return
            else:
                self.align_hold_count = 0
        elif self.phase == "ANGLE_ALIGN":
            self.publish_mode(self.MODE_DIFF)
            self.publish_phase("PREALIGN")
            if bool(self.p("require_mode_feedback")) and self.mode_feedback != self.MODE_DIFF:
                self.publish_stop("WAITING_DIFF_MODE")
                return
            command = compute_diff_angle_command(
                entrance_x=entrance_x,
                entrance_y=entrance_y,
                bed_yaw=yaw,
                target_bed_yaw=float(self.p("target_bed_yaw")),
                aligned_yaw=float(self.p("angle_align_yaw_tolerance")),
                max_angular=float(self.p("angle_align_max_angular")),
                min_angular=float(self.p("angle_align_min_angular")),
                k_yaw=float(self.p("angle_align_k_yaw")),
            )
            if command.state == "ANGLE_HOLD":
                self.angle_hold_count += 1
                if self.angle_hold_count >= int(self.p("angle_align_hold_frames")):
                    if bool(self.p("prealign_only")):
                        self.angle_hold_count = int(self.p("angle_align_hold_frames"))
                        self.publish_stop("PREALIGN_COMPLETE")
                        return
                    self.phase = "INSERT"
                    self.angle_hold_count = 0
                    axis_x, axis_y = cos(yaw), sin(yaw)
                    self.insert_start_along = axis_x * entrance_x + axis_y * entrance_y
                    self.publish_stop("ANGLE_ALIGNED")
                    return
            else:
                self.angle_hold_count = 0
        else:
            self.publish_mode(self.MODE_DIFF)
            current_along = cos(yaw) * entrance_x + sin(yaw) * entrance_y
            if self.insert_start_along is None:
                self.insert_start_along = current_along
            perception_phase = insertion_detection_phase(
                current_along=current_along,
                insert_start_along=self.insert_start_along,
                top_handoff_start_distance=float(
                    self.p("top_handoff_start_distance")
                ),
            )
            self.publish_phase(perception_phase)
            if perception_phase == "INSERT_FOOT":
                self.top_handoff_search_start_along = None
            elif bool(self.p("stop_on_top_handoff")):
                if self.top_handoff_search_start_along is None:
                    self.top_handoff_search_start_along = current_along
                handoff_result = evaluate_top_handoff_test(
                    detection_mode=self.detection_mode_feedback,
                    current_along=current_along,
                    insert_start_along=self.top_handoff_search_start_along,
                    max_insert_distance=float(
                        self.p("top_handoff_test_max_insert_distance")
                    ),
                )
                if handoff_result is not None:
                    self.phase = (
                        "TOP_HANDOFF_TEST_COMPLETE"
                        if handoff_result == "TOP_HANDOFF_COMPLETE"
                        else "TOP_HANDOFF_TEST_LIMIT"
                    )
                    self.publish_stop(handoff_result)
                    return
            # Never keep advancing on a clearly wrong RANSAC association.
            # DIFF may correct a small cross-track/yaw error while translating
            # along base Y; a large association error always stops the robot.
            target_bed_yaw = float(self.p("target_bed_yaw"))
            yaw_error = atan2(sin(yaw - target_bed_yaw), cos(yaw - target_bed_yaw))
            cross_error = -sin(yaw) * entrance_x + cos(yaw) * entrance_y
            cross_error -= float(self.p("target_cross"))
            if (
                abs(cross_error) > float(self.p("insert_abort_cross"))
                or abs(yaw_error) > float(self.p("insert_abort_yaw"))
            ):
                self.publish_stop("INSERT_ALIGNMENT_LOST")
                return
            command = compute_insert_command(
                entrance_x=entrance_x,
                entrance_y=entrance_y,
                bed_yaw=yaw,
                target_entrance_along=float(self.p("target_entrance_along")),
                target_cross=float(self.p("target_cross")),
                stop_along=float(self.p("stop_along")),
                stop_cross=float(self.p("stop_cross")),
                stop_yaw=float(self.p("stop_yaw")),
                max_linear=float(self.p("insert_max_speed")),
                max_reverse=float(self.p("insert_max_reverse_speed")),
                max_angular=float(self.p("insert_max_angular")),
                k_along=float(self.p("insert_k_along")),
                k_cross=float(self.p("insert_k_cross")),
                k_yaw=float(self.p("insert_k_yaw")),
                cross_deadband=float(self.p("insert_cross_deadband")),
                yaw_deadband=float(self.p("insert_yaw_deadband")),
                target_bed_yaw=target_bed_yaw,
            )
            if command.state == "DOCKED":
                self.phase = "DOCKED"
        twist = Twist()
        twist.linear.x = command.linear
        twist.linear.y = command.lateral
        twist.angular.z = command.angular
        self.cmd_pub.publish(twist)
        self.publish_state(command.state)

    def destroy_node(self):
        # A launch shutdown can invalidate the ROS context before this hook is
        # called. Publish the final safe state only while publishers are valid.
        if rclpy.ok():
            try:
                self.cmd_pub.publish(Twist())
                self.publish_mode(self.MODE_ACKERMANN)
                self.publish_phase("PREALIGN")
            except Exception:
                pass
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DockingControllerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.destroy_node()
        except (KeyboardInterrupt, Exception):
            pass
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception:
                pass


if __name__ == "__main__":
    main()
