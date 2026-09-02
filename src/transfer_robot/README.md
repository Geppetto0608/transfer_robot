# transfer_robot

ROS 2 Humble 기반 환자 이승 로봇의 실기 전용 도킹 패키지다. 3대 Livox Mid-360을
`base_link` 기준으로 병합한 `PointCloud2`에서 침대 입구와 상판 외곽선을 검출하고, 실제 4WS 하부 제어기에
`/cmd_vel`과 구동 모드 요청을 보낸다. 시뮬레이터 자산이나 시뮬레이션 실행 경로는
포함하지 않는다.

## 동작 구조

```text
3 x Mid-360 -> /livox_merge/merged_pointcloud
  -> TF로 base_link 변환
  -> ROI / 최근 병합 cloud 3프레임(최대 0.6초) 누적 / voxel
  -> radius outlier / clustering
  -> FOOT_RANSAC(사전 정렬)
  -> 상판 높이층 + 발판 pose 기반 회전 ROI
  -> TOP_OUTLINES(삽입)
  -> 침대 pose + confidence
  -> ACKERMANN 직선 이동으로 침대 판 중앙 정렬
  -> 완전 정지 후 DIFF 모드 전환 확인
  -> DIFF 제자리 회전으로 침대 각도 정렬
  -> DIFF로 침대 방향 저속 삽입
```

- `base_link +X`: 로봇 전방, ACKERMANN 좌우 정렬 이동축
- `base_link +Y`: 로봇 좌측
- `base_link -Y`: 로봇 우측, 현재 침대 진입 방향
- `base_link +Z`: 위쪽
- ACKERMANN 중앙 정렬: `linear.x`만 사용
- DIFF 모드: `linear.y`, `angular.z` 사용

현재 순서는 `ACKERMANN(코드 1) → 정지 → DIFF(코드 2)`다. ACKERMANN은
`linear.x` 직선 이동으로 침대 입구 가로판의 수직 중심선에 맞추고, 중앙 정렬이
연속 3프레임 확인된 뒤 0 속도를 5프레임 발행한다. 실제 DIFF 피드백이 확인되면
`angular.z`만 사용해 목표 yaw에 맞춘다. 전체 도킹에서는 그 다음
`linear.y < 0`으로 침대 안쪽에 진입한다.

PREALIGN에서는 근거점이 적은 뒤쪽 평행선을 제외하고 발판을 추적한다. INSERT가
시작되면 직전 발판 중심과 각도로 침대 방향의 회전된 동적 ROI를 만든다. 병합 cloud에서
INSERT 시작 후 0.20 m까지는 발판 높이층의 `FOOT_RANSAC`만 사용한다. 이후 상판
높이층에서 양쪽 긴 외곽선을 각각 RANSAC으로 추출해 폭과 평행도를 검증한다. 전환
순간 FOOT/TOP의 고정 pose offset을 보정하고 TOP 후보가 연속 3프레임 확인되면
`FOOT_RANSAC → TOP_OUTLINES`로 전환한다. 기존 `SIDE_RANSAC` 코드는 센서 진단과
보조 검출용으로 남아 있지만 AUTO 도킹의 주 전환 경로에는 사용하지 않는다.

상단·전방·후방 라이다가 같은 침대 면을 정확히 같은 시각에 채우지 않으므로 검출기는
단일 병합 메시지로 즉시 판단하지 않는다. `base_link` ROI에 들어온 최근 병합 cloud
3장을 최대 0.6초 동안 누적하고 3 cm voxel로 다시 합친 뒤 검출한다. 그 뒤에도 pose는
3회, TOP 전환은 3회 연속 확인해야 확정된다. 현재 저속에서는 누적 중 상대 이동으로
생기는 번짐이 대략 2~4 cm 범위다.

## Humble 빌드

대상 환경은 Ubuntu 22.04, ROS 2 Humble, Python 3.10이다.

```bash
cd /home/test/jun_ws
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select transfer_robot --symlink-install
source install/setup.bash
```

## 실기 실행

기본 입력은 `/livox_merge/merged_pointcloud`, 기준 좌표계는 `base_link`다.
먼저 bringup의 센서 계층을 실행한다.

```bash
source /home/test/test/install/setup.bash
ros2 launch bringup sensors.launch.py
```

다른 터미널에서 도킹 패키지를 실행한다.

```bash
ros2 launch transfer_robot transfer_robot.launch.py
```

병합 토픽 이름을 변경한 경우에만 launch 인자로 덮어쓴다.

```bash
ros2 launch transfer_robot transfer_robot.launch.py \
  pointcloud_topic:=/livox_merge/merged_pointcloud base_frame:=base_link
```

launch는 검출기, 도킹 제어기, `platform_mode_bridge`를 실행한다. 도킹 제어기는
`enabled: false`, `prealign_only: false`로 시작한다. 명시적으로 활성화하기 전까지
0 속도를 계속 발행한다. 활성화하면 ACKERMANN 중앙 정렬, DIFF 각도 정렬,
FOOT 0.20 m 삽입, TOP outline 삽입을 연속 수행한다. 현재 최종 거리는 작업자가
직접 정지하는 시험 설정이다.

launch 시작 1초 뒤 `/docking`에 `std_msgs/msg/Bool {data: true}`를 한 번 발행해
외부 `top_tf_switcher`의 상단 LiDAR TF를 도킹 위치로 전환한다.

삽입된 상태에서 저속으로 배출하려면 `RETRACT` 서비스를 사용한다. 서비스는 DIFF
피드백과 상판의 두 긴 외곽선(`TOP_OUTLINES`) pose를 확인한 뒤 `base_link +Y`
방향으로 이동하면서 중심선과 yaw를
동시에 보정한다. pose가 끊기거나 신뢰도가 낮거나 정렬 오차가 제한을 넘으면 정지하며,
자동 거리 종료는 하지 않는다. `retract false`는 DIFF와 검출 추적을 유지한 채 일시
정지하므로 같은 서비스에 `true`를 다시 보내 바로 재개할 수 있다. 아커만으로 완전히
복귀하려면 `/docking_controller/enable`에 `false`를 보낸다.

```bash
ros2 service call /docking_controller/retract std_srvs/srv/SetBool "{data: true}"
ros2 service call /docking_controller/retract std_srvs/srv/SetBool "{data: false}"
```

```bash
ros2 service call /docking_controller/enable std_srvs/srv/SetBool "{data: true}"
ros2 service call /docking_controller/enable std_srvs/srv/SetBool "{data: false}"
```

FOOT→TOP 전환에서 자동으로 멈추는 시험이 필요하면 먼저 제어기를 비활성화하고
`stop_on_top_handoff`를 켠 뒤 다시 활성화한다.

```bash
ros2 service call /docking_controller/enable std_srvs/srv/SetBool "{data: false}"
ros2 param set /docking_controller stop_on_top_handoff true
ros2 service call /docking_controller/enable std_srvs/srv/SetBool "{data: true}"
```

이 시험은 INSERT 시작 후 0.20 m까지 FOOT_RANSAC을 사용하고, TOP 검색 시작 후
추가 0.35 m 안에 전환하지 못하면 `TOP_HANDOFF_NOT_FOUND`로 정지한다. 정상 전환은
`TOP_HANDOFF_COMPLETE`로 정지한다. 연속 삽입 기본값으로 복귀하려면 다음처럼 보호를
다시 해제한다.

```bash
ros2 service call /docking_controller/enable std_srvs/srv/SetBool "{data: false}"
ros2 param set /docking_controller stop_on_top_handoff false
ros2 service call /docking_controller/enable std_srvs/srv/SetBool "{data: true}"
```

## 실기 시작 전 필수 점검

```bash
ros2 topic type /livox_merge/merged_pointcloud
ros2 topic hz /livox_merge/merged_pointcloud
ros2 topic hz /livox/lidar_192_168_1_135
ros2 topic hz /livox/lidar_192_168_2_102
ros2 topic hz /livox/lidar_192_168_3_144
ros2 run tf2_ros tf2_echo base_link livox_top
ros2 run tf2_ros tf2_echo base_link livox_front
ros2 run tf2_ros tf2_echo base_link livox_rear
ros2 topic echo /motor_node/drive_mode
ros2 topic echo /docking_controller/status
```

1. 세 원본 LiDAR와 병합 토픽 타입은 반드시 `sensor_msgs/msg/PointCloud2`여야 한다.
2. `livox_top`, `livox_front`, `livox_rear`에서 `base_link`까지 유효한 TF가 있어야 한다.
3. `/motor_node/drive_mode`가 실제 하부제어기 상태를 문자열로 발행해야 한다.
4. `/mode`의 실기 매핑이 `ACKERMANN=1`, `DIFF=2`인지 하부제어기와 대조한다.
5. `target_entrance_along`과 `target_cross`는 실제 플랫폼과 침대에서 다시 측정한다.
6. 첫 시험은 구동륜을 띄우거나 저속·비상정지 가능 상태에서 수행한다.

## ROS 인터페이스

| 구분 | 이름 | 타입/의미 |
|---|---|---|
| 입력 | `/livox_merge/merged_pointcloud` | 3대가 `base_link`로 변환된 `sensor_msgs/msg/PointCloud2` |
| 입력 | TF | 병합 노드에서 사용하는 `base_link <- livox_top/front/rear` |
| 입력 | `/motor_node/drive_mode` | 하부제어기 모드 상태 문자열 |
| 출력 | `/detected_dock_pose` | `base_link` 기준 침대 입구 pose |
| 출력 | `/bed_detection/confidence` | 검출 신뢰도 0~1 |
| 출력 | `/bed_detection/fusion_frames` | 현재 판단에 누적된 병합 cloud 수 |
| 출력 | `/bed_detection/ransac_inliers` | 선택된 기하 모델 근거점 |
| 출력 | `/bed_detection/top_roi` | 발판 pose와 상판 Z층으로 자른 동적 회전 ROI |
| 출력 | `/bed_detection/markers` | RViz 검출 마커 |
| 출력 | `/cmd_vel` | 플랫폼 속도 명령 |
| 출력 | `/drive_mode` | `DIFF` 또는 `ACKERMANN` 요청 |
| 출력 | `/mode` | 하부제어기 모드 코드, 1 또는 2 |
| 출력 | `/drive_mode_feedback` | 정규화된 모드 피드백 |
| 출력 | `/docking_controller/status` | 제어 상태 및 정지 원인 |
| 서비스 | `/bed_detector/enable` | 검출 활성화/비활성화 |
| 서비스 | `/docking_controller/enable` | 도킹 시작/정지 |
| 서비스 | `/docking_controller/retract` | 보정 배출 시작/일시정지 |

## 안전 동작

- 검출 pose가 1.5초 이상 끊기면 `LOST`로 정지한다. 현재 3-LiDAR 병합/RANSAC
  실측 처리율 약 1.3 Hz를 반영한 값이다.
- confidence가 기준 미만이면 `LOW_CONFIDENCE`로 정지한다.
- 실제 DIFF/ACKERMANN 피드백이 확인되지 않으면 진행하지 않는다.
- 모드 전환이 8초 안에 완료되지 않으면 `MODE_SWITCH_TIMEOUT`으로 정지한다.
- 삽입 중 횡오차 0.25 m 또는 yaw 오차 0.20 rad 초과 시
  `INSERT_ALIGNMENT_LOST`로 정지한다.
- 배출 중에도 침대 중심선/yaw를 보정하며 같은 제한을 넘으면
  `RETRACT_ALIGNMENT_LOST`로 정지한다.
- 비활성화/종료 시 0 속도와 ACKERMANN 복귀 요청을 발행한다.

이 정지는 소프트웨어 계층의 보호 기능이다. 하드웨어 비상 정지, 모터 드라이버
watchdog, 속도/조향 제한을 대체하지 않는다.

## 실기 튜닝 파일

- `config/bed_detector.yaml`: LiDAR ROI, 상판 높이층, FOOT/TOP 검출 및 추적 파라미터
- `config/docking_controller.yaml`: 속도 제한, 허용 오차, 최종 도킹 offset

특히 `roi_x/y/z_min/max`, `top_z_min/max`, `top_entrance_surface_offset`,
`foot_line_axis`, `target_entrance_along`은 3대 병합 후의 실제 침대 형상과 포인트 분포에
직접 의존한다. 현재 정렬 위치 실측값은 `foot_z: 0.22~0.38`,
`top_z: 0.54~0.70`이다. RViz Fixed
Frame을 `base_link`로 두고 `/bed_detection/top_roi` 및 `/clicked_point`로 실제 상판을
확인해 재조정하기 전에는 전체 거리 삽입을 수행하지 않는다.

## 테스트

```bash
cd /home/test/jun_ws
source /opt/ros/humble/setup.bash
colcon test --packages-select transfer_robot
colcon test-result --verbose
```
