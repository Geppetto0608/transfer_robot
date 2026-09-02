from __future__ import annotations

from collections import deque
from math import cos, sin

import numpy as np
import rclpy
from geometry_msgs.msg import Point, PoseStamped
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from scipy.spatial import cKDTree
from std_msgs.msg import Float32, Header, Int32, String
from std_srvs.srv import SetBool
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from .geometry import (
    BedDetection,
    apply_bed_pose_calibration,
    detect_bed_foot_line,
    detect_bed_side_lines,
    detect_bed_top_outlines,
    detect_bed_top_rectangle,
    euclidean_clusters,
    planar_downsample,
    radius_outlier_filter,
    voxel_downsample,
    wrap_angle,
)
from .pointcloud import read_xyz_points


# Safe standalone defaults. The hardware YAML overrides these values for the
# real platform; only live-platform configuration is installed.
DETECTOR_DEFAULTS: dict[str, object] = {
    # ROS interface
    "input_topic": "/point_cloud",
    "base_frame": "base_link",
    # AUTO follows /docking_phase. Forced modes support sensor-only diagnostics.
    "detection_mode": "AUTO",
    "enabled": True,
    # Full/filtered clouds are disabled on the live platform to limit bandwidth.
    "publish_debug_clouds": False,

    # ROI in base_frame [m]
    "roi_x_min": 0.15,
    "roi_x_max": 3.5,
    "roi_y_min": -1.5,
    "roi_y_max": 1.5,
    "roi_z_min": 0.08,
    "roi_z_max": 0.48,

    # Point-cloud filtering
    "voxel_leaf": 0.025,
    "outlier_radius": 0.10,
    "outlier_min_neighbors": 2,
    # Accumulate a few already-merged snapshots. The top/front/rear scanners
    # do not illuminate the same bed surface at exactly the same instant.
    "temporal_fusion_frames": 3,
    "temporal_fusion_max_age": 0.60,

    # Euclidean clustering
    "cluster_tolerance": 0.12,
    "cluster_min_points": 8,
    "cluster_max_points": 5000,

    # FOOT_RANSAC geometry
    "expected_foot_width": 0.95,
    "foot_width_tolerance": 0.30,
    "expected_bed_length": 2.04,
    "line_distance_threshold": 0.06,
    "line_min_points": 30,
    "line_min_observed_length": 0.55,
    "foot_line_axis": "Y",
    "foot_bed_axis_sign": 1.0,
    "foot_min_abs_direction": 0.90,
    "foot_tracking_position_weight": 4.0,
    "foot_tracking_yaw_weight": 2.5,
    "foot_z_min": 0.22,
    "foot_z_max": 0.38,

    # SIDE_RANSAC geometry and boundary extraction
    "side_expected_width": 0.90,
    "side_width_tolerance": 0.22,
    "side_min_observed_length": 0.60,
    "side_min_abs_direction_x": 0.90,
    "side_parallel_tolerance_deg": 12.0,
    "side_bin_width": 0.05,
    "side_bin_min_points": 8,
    "side_boundary_low_percentile": 3.0,
    "side_boundary_high_percentile": 97.0,
    "side_boundary_distance_threshold": 0.035,
    "side_boundary_min_bins": 6,
    # Collapse upper/lower returns into XY occupancy before SIDE fitting.
    "side_projection_leaf": 0.04,
    "side_symmetry_weight": 2.0,
    "side_envelope_preference": 0.35,
    "side_tracking_position_weight": 1.2,
    # Expected bed length/insertion heading in base_link. The real transfer
    # opening is on the robot's -Y side.
    "bed_axis_yaw": -1.5707963267948966,

    # TOP_RECTANGLE geometry. The height limits are provisional until the
    # actual mattress/top points are clicked in RViz in base_link.
    "top_z_min": 0.52,
    "top_z_max": 0.88,
    "top_expected_width": 0.90,
    "top_width_tolerance": 0.20,
    "top_length_tolerance": 0.45,
    "top_min_observed_length": 0.55,
    "top_min_points": 40,
    "top_max_yaw_delta": 0.26,
    "top_entrance_surface_offset": 0.0,
    "top_projection_leaf": 0.04,
    # Dynamic oriented ROI around the last FOOT/TOP entrance pose.
    "top_roi_behind": 0.25,
    "top_roi_ahead": 0.35,
    "top_roi_cross_margin": 0.20,
    "top_handoff_max_position_jump": 0.45,
    "top_handoff_max_yaw_jump": 0.30,
    "top_calibration_max_position_offset": 0.65,
    "top_calibration_max_yaw_offset": 0.12,
    "top_switch_confirm_frames": 3,
    "publish_top_roi": True,

    # Detection confirmation, tracking, and smoothing
    "min_confidence": 0.45,
    "confirm_frames": 3,
    "tracking_reset_frames": 10,
    "tracking_max_position_jump": 0.12,
    "tracking_max_yaw_jump": 0.22,
    "filter_alpha": 0.25,
}


class BedDetectorNode(Node):
    def __init__(self) -> None:
        super().__init__("bed_detector")
        for name, default in DETECTOR_DEFAULTS.items():
            self.declare_parameter(name, default)

        self.enabled = bool(self.get_parameter("enabled").value)
        requested_detection_mode = str(self.get_parameter("detection_mode").value).strip().upper()
        valid_detection_modes = (
            "AUTO",
            "FOOT_RANSAC",
            "TOP_OUTLINES",
            "TOP_RECTANGLE",  # Backward-compatible forced diagnostic mode.
            "SIDE_RANSAC",
        )
        if requested_detection_mode not in valid_detection_modes:
            raise ValueError(
                f"detection_mode must be one of {valid_detection_modes}, "
                f"got '{requested_detection_mode}'"
            )
        self.forced_detection_mode = requested_detection_mode
        self.foot_line_axis = str(self.get_parameter("foot_line_axis").value).strip().upper()
        if self.foot_line_axis not in ("X", "Y"):
            raise ValueError(
                f"foot_line_axis must be X or Y, got '{self.foot_line_axis}'"
            )
        self.hit_streak = 0
        self.miss_streak = 0
        self.filtered_pose: tuple[float, float, float] | None = None
        self.last_tf_warning_ns = 0
        self.docking_phase = "PREALIGN"
        self.detection_mode = (
            "FOOT_RANSAC"
            if self.forced_detection_mode == "AUTO"
            else self.forced_detection_mode
        )
        self.top_hit_streak = 0
        self.top_alignment: tuple[float, float, float] | None = None
        self.latest_top_roi = np.empty((0, 3), dtype=np.float32)
        history_size = max(1, int(self.p("temporal_fusion_frames")))
        self.cloud_history: deque[tuple[float, np.ndarray]] = deque(
            maxlen=history_size
        )
        self.fusion_frames_used = 0

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        debug_cloud_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
        )
        self.subscription = self.create_subscription(
            PointCloud2,
            str(self.get_parameter("input_topic").value),
            self.point_cloud_callback,
            sensor_qos,
        )
        self.phase_subscription = self.create_subscription(
            String, "/docking_phase", self.phase_callback, 10
        )
        self.pose_pub = self.create_publisher(PoseStamped, "/detected_dock_pose", 10)
        self.confidence_pub = self.create_publisher(Float32, "/bed_detection/confidence", 10)
        self.fusion_frames_pub = self.create_publisher(
            Int32, "/bed_detection/fusion_frames", 10
        )
        self.ransac_inliers_pub = self.create_publisher(
            PointCloud2, "/bed_detection/ransac_inliers", debug_cloud_qos
        )
        self.transformed_cloud_pub = None
        self.roi_cloud_pub = None
        self.top_roi_cloud_pub = None
        if bool(self.p("publish_debug_clouds")):
            self.transformed_cloud_pub = self.create_publisher(
                PointCloud2, "/bed_detection/transformed_cloud", debug_cloud_qos
            )
            self.roi_cloud_pub = self.create_publisher(
                PointCloud2, "/bed_detection/roi_cloud", debug_cloud_qos
            )
        if bool(self.p("publish_top_roi")):
            self.top_roi_cloud_pub = self.create_publisher(
                PointCloud2, "/bed_detection/top_roi", debug_cloud_qos
            )
        self.marker_pub = self.create_publisher(MarkerArray, "/bed_detection/markers", 10)
        self.mode_pub = self.create_publisher(String, "/bed_detection/mode", 10)
        self.enable_service = self.create_service(SetBool, "/bed_detector/enable", self.set_enabled)
        self.get_logger().info(
            f"Bed detector ready; input={self.p('input_topic')}, "
            f"base_frame={self.p('base_frame')}, mode={self.forced_detection_mode}"
        )

    def p(self, name: str):
        return self.get_parameter(name).value

    def set_enabled(self, request: SetBool.Request, response: SetBool.Response):
        self.enabled = bool(request.data)
        self.hit_streak = 0
        self.miss_streak = 0
        self.filtered_pose = None
        self.detection_mode = (
            "FOOT_RANSAC"
            if self.forced_detection_mode == "AUTO"
            else self.forced_detection_mode
        )
        self.top_hit_streak = 0
        self.top_alignment = None
        self.latest_top_roi = np.empty((0, 3), dtype=np.float32)
        self.cloud_history.clear()
        self.fusion_frames_used = 0
        response.success = True
        response.message = "bed detector enabled" if self.enabled else "bed detector disabled"
        return response

    def phase_callback(self, message: String) -> None:
        if self.forced_detection_mode != "AUTO":
            return
        phase = str(message.data).strip().upper()
        if phase not in (
            "PREALIGN",
            "INSERT_FOOT",
            "INSERT_TOP",
            "RETRACT",
            "DOCKED",
        ) or phase == self.docking_phase:
            return
        self.docking_phase = phase
        self.hit_streak = 0
        self.miss_streak = 0
        self.top_hit_streak = 0
        if phase != "RETRACT":
            self.top_alignment = None
        if phase == "PREALIGN":
            self.detection_mode = "FOOT_RANSAC"
            self.filtered_pose = None
            self.cloud_history.clear()
            self.fusion_frames_used = 0
        elif phase == "RETRACT":
            # The foot-board is normally outside the LiDAR view once the
            # platform is inserted. Retract therefore uses the bed-top pair
            # directly and preserves an existing TOP calibration/track.
            self.detection_mode = "TOP_OUTLINES"
        self.get_logger().info(f"Docking phase: {phase}; detector: {self.detection_mode}")

    @staticmethod
    def quaternion_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
        norm = x * x + y * y + z * z + w * w
        if norm < 1.0e-12:
            return np.eye(3)
        scale = 2.0 / norm
        return np.array(
            [
                [1.0 - scale * (y * y + z * z), scale * (x * y - z * w), scale * (x * z + y * w)],
                [scale * (x * y + z * w), 1.0 - scale * (x * x + z * z), scale * (y * z - x * w)],
                [scale * (x * z - y * w), scale * (y * z + x * w), 1.0 - scale * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    def transform_to_base(
        self, points: np.ndarray, source_frame: str, stamp
    ) -> np.ndarray | None:
        base_frame = str(self.p("base_frame"))
        if source_frame == base_frame:
            return points
        try:
            transform = self.tf_buffer.lookup_transform(
                base_frame,
                source_frame,
                rclpy.time.Time.from_msg(stamp),
                timeout=Duration(seconds=0.1),
            ).transform
        except TransformException as error:
            now = self.get_clock().now().nanoseconds
            if now - self.last_tf_warning_ns > 2_000_000_000:
                self.get_logger().warning(f"TF {base_frame} <- {source_frame} unavailable: {error}")
                self.last_tf_warning_ns = now
            return None
        q = transform.rotation
        rotation = self.quaternion_matrix(q.x, q.y, q.z, q.w)
        translation = np.array(
            [transform.translation.x, transform.translation.y, transform.translation.z], dtype=np.float64
        )
        return (points @ rotation.T + translation).astype(np.float32)

    def preprocess(self, points: np.ndarray, stamp_seconds: float) -> np.ndarray:
        finite = np.isfinite(points).all(axis=1)
        points = points[finite]
        mask = (
            (points[:, 0] >= float(self.p("roi_x_min")))
            & (points[:, 0] <= float(self.p("roi_x_max")))
            & (points[:, 1] >= float(self.p("roi_y_min")))
            & (points[:, 1] <= float(self.p("roi_y_max")))
            & (points[:, 2] >= float(self.p("roi_z_min")))
            & (points[:, 2] <= float(self.p("roi_z_max")))
        )
        points = voxel_downsample(points[mask], float(self.p("voxel_leaf")))

        # All samples are already expressed in base_link. At the commissioning
        # speed (<= 0.04 m/s, <= 0.05 rad/s), a 0.6 s window adds only a few cm
        # of bounded smear while filling complementary top/lower LiDAR views.
        if self.cloud_history and stamp_seconds < self.cloud_history[-1][0]:
            self.cloud_history.clear()
        self.cloud_history.append((stamp_seconds, points))
        max_age = max(0.0, float(self.p("temporal_fusion_max_age")))
        while (
            self.cloud_history
            and stamp_seconds - self.cloud_history[0][0] > max_age
        ):
            self.cloud_history.popleft()
        self.fusion_frames_used = len(self.cloud_history)
        if self.cloud_history:
            points = np.vstack([cloud for _, cloud in self.cloud_history])
            # Equalize repeat observations and sensor density after fusion.
            points = voxel_downsample(points, float(self.p("voxel_leaf")))
        return radius_outlier_filter(
            points,
            float(self.p("outlier_radius")),
            int(self.p("outlier_min_neighbors")),
        )

    def find_detection(self, points: np.ndarray) -> BedDetection | None:
        self.latest_top_roi = np.empty((0, 3), dtype=np.float32)
        clusters = euclidean_clusters(
            points[:, :2],
            float(self.p("cluster_tolerance")),
            int(self.p("cluster_min_points")),
            int(self.p("cluster_max_points")),
        )
        tracking_reference = None
        tracking_yaw = None
        if self.filtered_pose is not None:
            tracking_reference = np.asarray(self.filtered_pose[:2], dtype=np.float64)
            tracking_yaw = float(self.filtered_pose[2])

        def find_top() -> BedDetection | None:
            z_mask = (
                (points[:, 2] >= float(self.p("top_z_min")))
                & (points[:, 2] <= float(self.p("top_z_max")))
            )
            top_points = points[z_mask]
            if len(top_points) == 0:
                return None

            reacquiring = tracking_reference is None or tracking_yaw is None
            reference = (
                np.zeros(2, dtype=np.float64)
                if reacquiring
                else tracking_reference
            )
            reference_yaw = (
                float(self.p("bed_axis_yaw"))
                if reacquiring
                else float(tracking_yaw)
            )
            axis = np.array(
                [cos(reference_yaw), sin(reference_yaw)], dtype=np.float64
            )
            normal = np.array([-axis[1], axis[0]], dtype=np.float64)
            if reacquiring:
                # A process may restart while the robot is already under the
                # bed. Search the configured top Z/XY ROI without requiring a
                # now-invisible foot-board seed. The geometry fit still
                # requires the known width, two parallel outlines and nominal
                # bed direction.
                self.latest_top_roi = top_points
            else:
                relative = top_points[:, :2] - reference
                along = relative @ axis
                cross = relative @ normal
                half_width = 0.5 * float(self.p("top_expected_width"))
                dynamic_mask = (
                    (along >= -float(self.p("top_roi_behind")))
                    & (
                        along
                        <= float(self.p("expected_bed_length"))
                        + float(self.p("top_roi_ahead"))
                    )
                    & (
                        np.abs(cross)
                        <= half_width + float(self.p("top_roi_cross_margin"))
                    )
                )
                self.latest_top_roi = top_points[dynamic_mask]
            if len(self.latest_top_roi) == 0:
                return None
            return detect_bed_top_outlines(
                self.latest_top_roi[:, :2],
                reference_entrance=reference,
                reference_yaw=reference_yaw,
                expected_width=float(self.p("top_expected_width")),
                width_tolerance=float(self.p("top_width_tolerance")),
                bed_length=float(self.p("expected_bed_length")),
                distance_threshold=float(self.p("line_distance_threshold")),
                min_points=int(self.p("line_min_points")),
                min_observed_length=float(self.p("top_min_observed_length")),
                min_abs_direction=float(self.p("foot_min_abs_direction")),
                parallel_tolerance_deg=float(self.p("side_parallel_tolerance_deg")),
                tracking_position_weight=(
                    0.0
                    if reacquiring
                    else float(self.p("side_tracking_position_weight"))
                ),
            )
        def find_side() -> BedDetection | None:
            # The geometry routine is expressed in a local frame where bed
            # length is +X. Rotate the merged base_link cloud into that frame,
            # then rotate the resulting pose/evidence back to base_link.
            bed_axis_yaw = float(self.p("bed_axis_yaw"))
            axis = np.array([cos(bed_axis_yaw), sin(bed_axis_yaw)], dtype=np.float64)
            normal = np.array([-axis[1], axis[0]], dtype=np.float64)
            basis = np.vstack((axis, normal))
            local_points = points[:, :2] @ basis.T
            # Upper and lower frame returns may be vertically thick. SIDE is
            # a planar fit, so give every occupied XY cell equal weight rather
            # than allowing the denser Z structure to dominate the result.
            local_points = planar_downsample(
                local_points, float(self.p("side_projection_leaf"))
            )
            side_clusters = euclidean_clusters(
                local_points,
                float(self.p("cluster_tolerance")),
                int(self.p("cluster_min_points")),
                int(self.p("cluster_max_points")),
            )
            local_tracking_reference = (
                None if tracking_reference is None else tracking_reference @ basis.T
            )
            local_tracking_yaw = (
                None if tracking_yaw is None else wrap_angle(tracking_yaw - bed_axis_yaw)
            )
            detection = detect_bed_side_lines(
                local_points,
                side_clusters,
                expected_width=float(self.p("side_expected_width")),
                width_tolerance=float(self.p("side_width_tolerance")),
                bed_length=float(self.p("expected_bed_length")),
                distance_threshold=float(self.p("line_distance_threshold")),
                min_points=int(self.p("line_min_points")),
                min_observed_length=float(self.p("side_min_observed_length")),
                min_abs_direction_x=float(self.p("side_min_abs_direction_x")),
                parallel_tolerance_deg=float(self.p("side_parallel_tolerance_deg")),
                bin_width=float(self.p("side_bin_width")),
                bin_min_points=int(self.p("side_bin_min_points")),
                boundary_low_percentile=float(self.p("side_boundary_low_percentile")),
                boundary_high_percentile=float(self.p("side_boundary_high_percentile")),
                boundary_distance_threshold=float(self.p("side_boundary_distance_threshold")),
                boundary_min_bins=int(self.p("side_boundary_min_bins")),
                symmetry_weight=float(self.p("side_symmetry_weight")),
                envelope_preference=float(self.p("side_envelope_preference")),
                tracking_position_weight=float(
                    self.p("side_tracking_position_weight")
                ),
                tracking_reference=local_tracking_reference,
                tracking_yaw=local_tracking_yaw,
            )
            if detection is None:
                return None

            def to_base(values: np.ndarray) -> np.ndarray:
                return values @ basis

            return BedDetection(
                entrance_center=to_base(detection.entrance_center),
                far_center=to_base(detection.far_center),
                yaw=wrap_angle(detection.yaw + bed_axis_yaw),
                confidence=detection.confidence,
                corners=to_base(detection.corners),
                width=detection.width,
                length=detection.length,
                evidence_points=to_base(detection.evidence_points),
            )

        def find_foot() -> BedDetection | None:
            foot_mask = (
                (points[:, 2] >= float(self.p("foot_z_min")))
                & (points[:, 2] <= float(self.p("foot_z_max")))
            )
            foot_points = points[foot_mask]
            if len(foot_points) == 0:
                return None
            foot_clusters = euclidean_clusters(
                foot_points[:, :2],
                float(self.p("cluster_tolerance")),
                int(self.p("cluster_min_points")),
                int(self.p("cluster_max_points")),
            )
            return detect_bed_foot_line(
                foot_points[:, :2], foot_clusters,
                expected_width=float(self.p("expected_foot_width")),
                width_tolerance=float(self.p("foot_width_tolerance")),
                bed_length=float(self.p("expected_bed_length")),
                distance_threshold=float(self.p("line_distance_threshold")),
                min_points=int(self.p("line_min_points")),
                min_observed_length=float(self.p("line_min_observed_length")),
                line_axis=self.foot_line_axis,
                min_abs_direction=float(self.p("foot_min_abs_direction")),
                tracking_reference=tracking_reference, tracking_yaw=tracking_yaw,
                tracking_position_weight=float(
                    self.p("foot_tracking_position_weight")
                ),
                tracking_yaw_weight=float(self.p("foot_tracking_yaw_weight")),
                bed_axis_sign=float(self.p("foot_bed_axis_sign")),
            )

        def calibrated_top(
            top: BedDetection | None,
        ) -> tuple[BedDetection | None, tuple[float, float, float] | None]:
            if top is None:
                return None, None
            if self.filtered_pose is None:
                if self.top_alignment is None:
                    # Standalone TOP reacquisition has no visible FOOT pose
                    # from which to estimate the model offset. The directly
                    # fitted centreline and yaw are sufficient for retract.
                    return top, None
                return (
                    apply_bed_pose_calibration(top, *self.top_alignment),
                    self.top_alignment,
                )
            reference = np.asarray(self.filtered_pose[:2], dtype=np.float64)
            reference_yaw = float(self.filtered_pose[2])
            calibration = self.top_alignment
            if calibration is None:
                delta = reference - top.entrance_center
                raw_axis = np.array([cos(top.yaw), sin(top.yaw)], dtype=np.float64)
                raw_normal = np.array([-raw_axis[1], raw_axis[0]], dtype=np.float64)
                position_offset = float(np.linalg.norm(delta))
                yaw_offset = wrap_angle(reference_yaw - top.yaw)
                if (
                    position_offset
                    > float(self.p("top_calibration_max_position_offset"))
                    or abs(yaw_offset)
                    > float(self.p("top_calibration_max_yaw_offset"))
                ):
                    return None, None
                calibration = (
                    float(np.dot(delta, raw_axis)),
                    float(np.dot(delta, raw_normal)),
                    yaw_offset,
                )
            return apply_bed_pose_calibration(top, *calibration), calibration

        if self.detection_mode == "SIDE_RANSAC":
            return find_side()
        if self.detection_mode in ("TOP_OUTLINES", "TOP_RECTANGLE"):
            top, pending_alignment = calibrated_top(find_top())
            if self.top_alignment is None and pending_alignment is not None:
                self.top_alignment = pending_alignment
            return top

        foot = find_foot()
        if self.forced_detection_mode == "FOOT_RANSAC":
            return foot
        if self.docking_phase not in ("INSERT_TOP", "RETRACT"):
            self.top_hit_streak = 0
            return foot

        # During INSERT, use the last FOOT pose to seed a bed-oriented top ROI.
        # This starts before the foot-board leaves the LiDAR field of view, so
        # the independently fitted top outlines can take over without a gap.
        top, pending_alignment = calibrated_top(find_top())
        if top is not None and self.filtered_pose is not None:
            foot_reference = np.asarray(self.filtered_pose[:2], dtype=np.float64)
            position_jump = float(np.linalg.norm(top.entrance_center - foot_reference))
            yaw_jump = abs(wrap_angle(top.yaw - float(self.filtered_pose[2])))
            if (
                position_jump > float(self.p("top_handoff_max_position_jump"))
                or yaw_jump > float(self.p("top_handoff_max_yaw_jump"))
            ):
                top = None
        if top is None or top.confidence < float(self.p("min_confidence")):
            self.top_hit_streak = 0
            return foot
        self.top_hit_streak += 1
        if self.top_hit_streak < int(self.p("top_switch_confirm_frames")):
            return foot

        self.detection_mode = "TOP_OUTLINES"
        self.top_alignment = pending_alignment
        # The handoff constraints above explicitly validate this one allowed
        # model change. Seed the filter with TOP so the ordinary small-jump
        # tracker does not reject a calibrated foot-to-surface offset.
        self.filtered_pose = (
            float(top.entrance_center[0]),
            float(top.entrance_center[1]),
            float(top.yaw),
        )
        self.hit_streak = 0
        self.miss_streak = 0
        self.get_logger().info("Bed detection handoff: FOOT_RANSAC -> TOP_OUTLINES")
        return top

    def is_tracking_consistent(self, detection: BedDetection) -> bool:
        if self.filtered_pose is None:
            return True
        previous_xy = np.asarray(self.filtered_pose[:2], dtype=np.float64)
        position_jump = float(np.linalg.norm(detection.entrance_center - previous_xy))
        yaw_jump = abs(wrap_angle(detection.yaw - float(self.filtered_pose[2])))
        return (
            position_jump <= float(self.p("tracking_max_position_jump"))
            and yaw_jump <= float(self.p("tracking_max_yaw_jump"))
        )

    def filter_detection(self, detection: BedDetection) -> tuple[float, float, float]:
        measurement = (float(detection.entrance_center[0]), float(detection.entrance_center[1]), detection.yaw)
        if self.filtered_pose is None:
            self.filtered_pose = measurement
            return measurement
        alpha = float(self.p("filter_alpha"))
        old_x, old_y, old_yaw = self.filtered_pose
        yaw = wrap_angle(old_yaw + alpha * wrap_angle(measurement[2] - old_yaw))
        self.filtered_pose = (
            old_x + alpha * (measurement[0] - old_x),
            old_y + alpha * (measurement[1] - old_y),
            yaw,
        )
        return self.filtered_pose

    def publish_pose(self, stamp, pose: tuple[float, float, float]) -> None:
        x, y, yaw = pose
        message = PoseStamped()
        message.header.stamp = stamp
        message.header.frame_id = str(self.p("base_frame"))
        message.pose.position.x = x
        message.pose.position.y = y
        message.pose.orientation.z = sin(0.5 * yaw)
        message.pose.orientation.w = cos(0.5 * yaw)
        self.pose_pub.publish(message)

    def publish_ransac_inliers(
        self,
        stamp,
        filtered_points: np.ndarray,
        detection: BedDetection | None,
    ) -> None:
        """Publish only the points supporting the selected RANSAC model.

        Geometry fitting is performed in XY. Map each selected XY inlier back
        to its nearest filtered 3D return so RViz still shows the real Z value.
        A geometrically valid candidate is shown even when confidence or
        tracking rejects it later, which makes false associations diagnosable.
        """
        header = Header(stamp=stamp, frame_id=str(self.p("base_frame")))
        if (
            detection is None
            or len(detection.evidence_points) == 0
            or len(filtered_points) == 0
        ):
            evidence_xyz = np.empty((0, 3), dtype=np.float32)
        else:
            tree = cKDTree(filtered_points[:, :2])
            _, nearest = tree.query(detection.evidence_points, k=1)
            evidence_xyz = filtered_points[np.asarray(nearest, dtype=np.int64)]
        self.ransac_inliers_pub.publish(
            point_cloud2.create_cloud_xyz32(header, evidence_xyz.tolist())
        )

    def publish_debug_cloud(self, publisher, stamp, points: np.ndarray) -> None:
        """Publish an XYZ cloud in base_frame when offline diagnostics are enabled."""
        if publisher is None:
            return
        header = Header(stamp=stamp, frame_id=str(self.p("base_frame")))
        publisher.publish(point_cloud2.create_cloud_xyz32(header, points.tolist()))

    def publish_markers(self, stamp, detection: BedDetection | None) -> None:
        delete = Marker()
        delete.header.stamp = stamp
        delete.header.frame_id = str(self.p("base_frame"))
        delete.action = Marker.DELETEALL
        markers = [delete]
        if detection is not None:
            legs = Marker()
            legs.header = delete.header
            legs.ns = "bed_frame_corners"
            legs.id = 1
            legs.type = Marker.SPHERE_LIST
            legs.action = Marker.ADD
            legs.scale.x = legs.scale.y = legs.scale.z = 0.10
            legs.color.r, legs.color.g, legs.color.b, legs.color.a = 0.1, 1.0, 0.2, 1.0
            legs.points = [Point(x=float(p[0]), y=float(p[1]), z=0.20) for p in detection.corners]

            lines = Marker()
            lines.header = delete.header
            lines.ns = "bed_geometry"
            lines.id = 2
            lines.type = Marker.LINE_LIST
            lines.action = Marker.ADD
            lines.scale.x = 0.025
            lines.color.r, lines.color.g, lines.color.b, lines.color.a = 1.0, 0.7, 0.0, 1.0
            a, b, c, d = detection.corners
            for p, q in ((a, b), (c, d), (a, c), (b, d)):
                lines.points.extend(
                    [Point(x=float(p[0]), y=float(p[1]), z=0.22), Point(x=float(q[0]), y=float(q[1]), z=0.22)]
                )

            arrow = Marker()
            arrow.header = delete.header
            arrow.ns = "bed_pose"
            arrow.id = 3
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD
            arrow.scale.x, arrow.scale.y, arrow.scale.z = 0.45, 0.07, 0.10
            arrow.color.r, arrow.color.g, arrow.color.b, arrow.color.a = 0.1, 0.5, 1.0, 1.0
            arrow.pose.position.x = float(detection.entrance_center[0])
            arrow.pose.position.y = float(detection.entrance_center[1])
            arrow.pose.position.z = 0.25
            arrow.pose.orientation.z = sin(0.5 * detection.yaw)
            arrow.pose.orientation.w = cos(0.5 * detection.yaw)
            markers.extend((legs, lines, arrow))
        self.marker_pub.publish(MarkerArray(markers=markers))

    def point_cloud_callback(self, message: PointCloud2) -> None:
        if not self.enabled:
            return
        try:
            points = read_xyz_points(message)
        except (AssertionError, ValueError) as error:
            self.get_logger().error(f"Cannot decode PointCloud2: {error}")
            return
        points = self.transform_to_base(points, message.header.frame_id, message.header.stamp)
        if points is None:
            return
        self.publish_debug_cloud(self.transformed_cloud_pub, message.header.stamp, points)
        stamp_seconds = (
            float(message.header.stamp.sec)
            + 1.0e-9 * float(message.header.stamp.nanosec)
        )
        if stamp_seconds <= 0.0:
            stamp_seconds = 1.0e-9 * float(self.get_clock().now().nanoseconds)
        filtered = self.preprocess(points, stamp_seconds)
        self.fusion_frames_pub.publish(Int32(data=self.fusion_frames_used))
        # This is the exact cloud presented to clustering/RANSAC: ROI, voxel,
        # and radius-outlier filtering have all already been applied.
        self.publish_debug_cloud(self.roi_cloud_pub, message.header.stamp, filtered)
        self.mode_pub.publish(String(data=self.detection_mode))
        detection = self.find_detection(filtered) if len(filtered) else None
        self.publish_debug_cloud(
            self.top_roi_cloud_pub, message.header.stamp, self.latest_top_roi
        )
        self.publish_ransac_inliers(message.header.stamp, filtered, detection)

        if detection is not None and not self.is_tracking_consistent(detection):
            detection = None

        if detection is None or detection.confidence < float(self.p("min_confidence")):
            self.hit_streak = 0
            self.miss_streak += 1
            if self.miss_streak >= int(self.p("tracking_reset_frames")):
                self.filtered_pose = None
            self.confidence_pub.publish(Float32(data=0.0))
            self.publish_markers(message.header.stamp, None)
            return

        self.miss_streak = 0
        self.hit_streak += 1
        confirmed_ratio = min(1.0, self.hit_streak / max(1, int(self.p("confirm_frames"))))
        confidence = float(detection.confidence * confirmed_ratio)
        self.confidence_pub.publish(Float32(data=confidence))
        self.publish_markers(message.header.stamp, detection)
        pose = self.filter_detection(detection)
        if self.hit_streak >= int(self.p("confirm_frames")):
            self.publish_pose(message.header.stamp, pose)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BedDetectorNode()
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
