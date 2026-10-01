"""Top-LiDAR wall detector used as the final insertion safety stop."""

from __future__ import annotations

from collections import deque
from math import atan2, cos, isfinite, sin
from time import monotonic

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2, Range
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker

from .pointcloud import read_xyz_points
from .wall_detection import detect_transverse_wall


DEFAULTS = {
    "input_topic": "/livox/lidar_192_168_1_135",
    "base_frame": "base_link",
    "bed_axis_yaw": -1.57079632679,
    "processing_interval": 0.25,
    "point_stride": 3,
    "min_distance": 0.80,
    "max_distance": 4.30,
    "corridor_half_width": 1.00,
    "z_min": 2.00,
    "z_max": 3.00,
    "bin_size": 0.05,
    "min_points": 100,
    "min_vertical_span": 0.01,
    "min_cross_span": 0.80,
    "height_bin_size": 0.10,
    "min_occupied_height_bins": 1,
    "height_bin_min_points": 2,
    "low_height_limit": 3.00,
    "min_low_height_fraction": 0.0,
    "peak_neighbor_bins": 3,
    "min_peak_ratio": 1.50,
    "confirm_frames": 3,
    "max_frame_jump": 0.15,
}


def quaternion_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    norm = x * x + y * y + z * z + w * w
    if norm <= 1.0e-12:
        return np.eye(3, dtype=np.float64)
    scale = 2.0 / norm
    return np.array(
        [
            [1.0 - scale * (y * y + z * z), scale * (x * y - z * w), scale * (x * z + y * w)],
            [scale * (x * y + z * w), 1.0 - scale * (x * x + z * z), scale * (y * z - x * w)],
            [scale * (x * z - y * w), scale * (y * z + x * w), 1.0 - scale * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


class WallDetectorNode(Node):
    def __init__(self) -> None:
        super().__init__("wall_detector")
        for name, default in DEFAULTS.items():
            self.declare_parameter(name, default)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        history_size = max(1, int(self.p("confirm_frames")))
        self.history: deque[float] = deque(maxlen=history_size)
        self.last_processing_time = -float("inf")

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        debug_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
        )
        self.create_subscription(
            PointCloud2,
            str(self.p("input_topic")),
            self.cloud_callback,
            sensor_qos,
        )
        self.range_pub = self.create_publisher(Range, "/bed_detection/wall_range", 10)
        self.points_pub = self.create_publisher(
            PointCloud2, "/bed_detection/wall_points", debug_qos
        )
        self.marker_pub = self.create_publisher(
            Marker, "/bed_detection/wall_marker", 10
        )
        self.get_logger().info(
            f"Wall detector ready; top_cloud={self.p('input_topic')}, "
            f"base_frame={self.p('base_frame')}, "
            f"interval={float(self.p('processing_interval')):.2f}s, "
            f"stride={int(self.p('point_stride'))}"
        )

    def p(self, name: str):
        return self.get_parameter(name).value

    def transform_to_base(
        self, points: np.ndarray, source_frame: str
    ) -> np.ndarray | None:
        target = str(self.p("base_frame"))
        if not source_frame or source_frame == target:
            return points
        try:
            transform = self.tf_buffer.lookup_transform(target, source_frame, Time())
        except TransformException as error:
            self.get_logger().warning(
                f"Wall cloud TF {source_frame}->{target} unavailable: {error}",
                throttle_duration_sec=2.0,
            )
            return None
        q = transform.transform.rotation
        t = transform.transform.translation
        rotation = quaternion_matrix(q.x, q.y, q.z, q.w)
        translation = np.array([t.x, t.y, t.z], dtype=np.float64)
        return (points.astype(np.float64) @ rotation.T + translation).astype(
            np.float32
        )

    def publish_range(self, stamp, distance: float) -> None:
        message = Range()
        message.header.stamp = stamp
        message.header.frame_id = str(self.p("base_frame"))
        message.radiation_type = Range.INFRARED
        message.field_of_view = 2.0 * atan2(
            float(self.p("corridor_half_width")), max(distance, 0.01)
        ) if isfinite(distance) else 0.0
        message.min_range = float(self.p("min_distance"))
        message.max_range = float(self.p("max_distance"))
        message.range = float(distance)
        self.range_pub.publish(message)

    def publish_marker(self, stamp, distance: float, visible: bool) -> None:
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = str(self.p("base_frame"))
        marker.ns = "wall_stop"
        marker.id = 0
        if not visible:
            marker.action = Marker.DELETE
            self.marker_pub.publish(marker)
            return
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.04
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = (
            1.0,
            0.1,
            0.1,
            1.0,
        )
        yaw = float(self.p("bed_axis_yaw"))
        axis = np.array([cos(yaw), sin(yaw)])
        normal = np.array([-axis[1], axis[0]])
        center = distance * axis
        half = float(self.p("corridor_half_width")) * normal
        low = float(self.p("z_min"))
        high = float(self.p("z_max"))
        a = center - half
        b = center + half
        marker.points = [
            Point(x=float(a[0]), y=float(a[1]), z=low),
            Point(x=float(b[0]), y=float(b[1]), z=low),
            Point(x=float(b[0]), y=float(b[1]), z=low),
            Point(x=float(b[0]), y=float(b[1]), z=high),
            Point(x=float(b[0]), y=float(b[1]), z=high),
            Point(x=float(a[0]), y=float(a[1]), z=high),
            Point(x=float(a[0]), y=float(a[1]), z=high),
            Point(x=float(a[0]), y=float(a[1]), z=low),
        ]
        self.marker_pub.publish(marker)

    def cloud_callback(self, message: PointCloud2) -> None:
        now = monotonic()
        if now - self.last_processing_time < float(self.p("processing_interval")):
            return
        self.last_processing_time = now

        try:
            points = read_xyz_points(message)
        except (AssertionError, ValueError) as error:
            self.get_logger().error(f"Cannot decode top PointCloud2: {error}")
            return
        # Wall stopping needs the range of a large plane, not every raw return.
        # Subsample before TF and geometry processing to keep bed detection from
        # being starved by the dense 10 Hz top-LiDAR cloud.
        stride = max(1, int(self.p("point_stride")))
        points = points[::stride]
        points = self.transform_to_base(points, message.header.frame_id)
        if points is None:
            return
        detection = detect_transverse_wall(
            points,
            axis_yaw=float(self.p("bed_axis_yaw")),
            min_distance=float(self.p("min_distance")),
            max_distance=float(self.p("max_distance")),
            corridor_half_width=float(self.p("corridor_half_width")),
            z_min=float(self.p("z_min")),
            z_max=float(self.p("z_max")),
            bin_size=float(self.p("bin_size")),
            min_points=int(self.p("min_points")),
            min_vertical_span=float(self.p("min_vertical_span")),
            min_cross_span=float(self.p("min_cross_span")),
            height_bin_size=float(self.p("height_bin_size")),
            min_occupied_height_bins=int(self.p("min_occupied_height_bins")),
            height_bin_min_points=int(self.p("height_bin_min_points")),
            low_height_limit=float(self.p("low_height_limit")),
            min_low_height_fraction=float(self.p("min_low_height_fraction")),
            peak_neighbor_bins=int(self.p("peak_neighbor_bins")),
            min_peak_ratio=float(self.p("min_peak_ratio")),
        )
        if detection is None:
            self.history.clear()
            self.publish_range(message.header.stamp, float("inf"))
            self.publish_marker(message.header.stamp, 0.0, False)
            return

        if self.history and abs(detection.distance - self.history[-1]) > float(
            self.p("max_frame_jump")
        ):
            self.history.clear()
        self.history.append(detection.distance)
        if len(self.history) < max(1, int(self.p("confirm_frames"))):
            self.publish_range(message.header.stamp, float("inf"))
            return

        distance = float(np.median(np.asarray(self.history)))
        self.publish_range(message.header.stamp, distance)
        header = Header(stamp=message.header.stamp, frame_id=str(self.p("base_frame")))
        self.points_pub.publish(
            point_cloud2.create_cloud_xyz32(header, detection.points.tolist())
        )
        self.publish_marker(message.header.stamp, distance, True)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WallDetectorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
