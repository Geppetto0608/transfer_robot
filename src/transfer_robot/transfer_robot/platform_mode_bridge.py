"""Translate generic docking mode requests to the real motor_node interface."""

from __future__ import annotations

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Int32, String


MODE_CODES = {"ACKERMANN": 1, "DIFF": 2}


def normalize_motor_status(status: str) -> str:
    """Return the active/pending hardware mode from motor_node's status text."""
    value = status.strip().upper()
    # A failed status can still contain the last requested mode, for example
    # ``FAILED|[DIFF] Motor feedback stale``.  Failure must take precedence;
    # otherwise the controller may command motion with unusable motor feedback.
    if value.startswith("FAILED|"):
        return "FAILED"
    if "[ACKERMANN]" in value or value.startswith("AUTONOMOUS|"):
        return "ACKERMANN"
    if "[DIFF]" in value or value.startswith("DOCKING|"):
        return "DIFF"
    if "[TO_ACKERMANN]" in value or value.startswith("EXITING_DOCKING|"):
        return "TO_ACKERMANN"
    if "[TO_DIFF]" in value or value.startswith("ENTERING_DOCKING|"):
        return "TO_DIFF"
    if "[ALIGN]" in value or value.startswith("ALIGN|"):
        return "ALIGN"
    return "UNKNOWN"


class PlatformModeBridge(Node):
    """Keep platform-specific numeric modes out of the docking core."""

    def __init__(self) -> None:
        super().__init__("platform_mode_bridge")
        self.declare_parameter("request_topic", "/drive_mode")
        self.declare_parameter("hardware_mode_topic", "/mode")
        self.declare_parameter("hardware_status_topic", "/motor_node/drive_mode")
        self.declare_parameter("feedback_topic", "/drive_mode_feedback")

        feedback_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        hardware_status_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            # motor_node publishes a latched state and may not publish again
            # until its mode changes. Receive that last state even when this
            # bridge starts after motor initialization.
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.mode_pub = self.create_publisher(
            Int32, str(self.get_parameter("hardware_mode_topic").value), 10
        )
        self.feedback_pub = self.create_publisher(
            String, str(self.get_parameter("feedback_topic").value), feedback_qos
        )
        self.create_subscription(
            String,
            str(self.get_parameter("request_topic").value),
            self.request_callback,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("hardware_status_topic").value),
            self.status_callback,
            hardware_status_qos,
        )
        self.last_request: str | None = None
        self.get_logger().info("Platform mode bridge ready: ACKERMANN=1, DIFF=2")

    def request_callback(self, message: String) -> None:
        requested = message.data.strip().upper()
        code = MODE_CODES.get(requested)
        if code is None:
            self.get_logger().warning(
                f"Ignoring unsupported drive mode '{message.data}'",
                throttle_duration_sec=2.0,
            )
            return
        # The docking controller publishes its state continuously. motor_node
        # should receive only transitions, not a new mode request every tick.
        if requested == self.last_request:
            return
        self.last_request = requested
        self.mode_pub.publish(Int32(data=code))
        self.get_logger().info(f"Requested motor mode {requested} ({code})")

    def status_callback(self, message: String) -> None:
        normalized = normalize_motor_status(message.data)
        if normalized in ("ALIGN", "FAILED"):
            # Permit a fresh request after motor_node initialization/recovery.
            self.last_request = None
        self.feedback_pub.publish(String(data=normalized))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PlatformModeBridge()
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
