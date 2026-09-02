# Patient Transfer Robot — ROS 2 Humble

환자 이승(transfer) 로봇의 **실기 전용 침대 도킹** 워크스페이스다.
3대의 Livox Mid-360을 `base_link` 기준으로 병합한 `PointCloud2`에서 침대 입구와
상판 외곽선을 검출하고, 실제 4WS(4-Wheel Steering) 하부 제어기에 `/cmd_vel`과
구동 모드를 요청해 침대 아래로 진입한다.

Isaac Sim, USD 모델, 시뮬레이터 드라이버, 시뮬레이션 launch는 포함하지 않는다.
메인 패키지는 [src/transfer_robot](src/transfer_robot)이다.

> **안전 고지** — 이 코드는 실제 이동형 로봇을 구동한다. 첫 시험은 반드시 구동륜을
> 띄우거나 저속·비상정지 가능한 상태에서 수행할 것. 의료기기로 승인된 소프트웨어가 아니다.

---

## 1. 요구 사항

| 항목 | 버전 / 사양 |
|---|---|
| OS | Ubuntu 22.04 |
| ROS | ROS 2 Humble |
| Python | 3.10 |
| 센서 | Livox Mid-360 × 3 (top / front / rear) |
| 플랫폼 | 4WS 하부 제어기 (ACKERMANN=1, DIFF=2) |

입력 토픽은 `/livox_merge/merged_pointcloud`, 기준 좌표계는 `base_link`다.
LiDAR 병합과 TF 발행은 별도의 bringup 패키지가 담당한다.

## 2. 저장소 구조

```text
.
├── README.md                      # 본 문서 (개요)
├── LICENSE                        # 연구 목적 한정 / 상업적 이용 금지
└── src/transfer_robot/
    ├── README.md                  # 상세 매뉴얼 (인터페이스·튜닝·점검표)
    ├── config/
    │   ├── bed_detector.yaml      # ROI, 상판 높이층, FOOT/TOP 검출 파라미터
    │   └── docking_controller.yaml# 속도 제한, 허용 오차, 도킹 offset
    ├── launch/transfer_robot.launch.py
    ├── transfer_robot/
    │   ├── node.py                # 침대 검출 노드
    │   ├── geometry.py            # RANSAC / 클러스터링 / 외곽선 기하
    │   ├── docking_controller_node.py
    │   ├── docking_control.py     # 도킹 상태 기계
    │   ├── platform_mode_bridge.py# 구동 모드 요청 ↔ 하부제어기 코드 변환
    │   └── pointcloud.py
    └── test/                      # colcon test 대상 단위 테스트
```

## 3. 빌드

```bash
cd ~/jun_ws
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select transfer_robot --symlink-install
source install/setup.bash
```

## 4. 실행

먼저 bringup의 센서 계층(3-LiDAR 드라이버 + 병합 + TF)을 띄운다.

```bash
ros2 launch bringup sensors.launch.py
```

다른 터미널에서 도킹 패키지를 실행한다. launch는 검출기, 도킹 제어기,
`platform_mode_bridge`를 함께 띄운다.

```bash
ros2 launch transfer_robot transfer_robot.launch.py
```

병합 토픽 이름이나 기준 좌표계를 바꾼 경우에만 인자로 덮어쓴다.

```bash
ros2 launch transfer_robot transfer_robot.launch.py \
  pointcloud_topic:=/livox_merge/merged_pointcloud base_frame:=base_link
```

**제어기는 항상 비활성 상태로 시작한다.** LiDAR, TF, 모터 모드 피드백, 비상 정지
장치를 모두 확인한 뒤에만 활성화한다.

```bash
# 도킹 시작
ros2 service call /docking_controller/enable std_srvs/srv/SetBool "{data: true}"

# 정지 (ACKERMANN 복귀)
ros2 service call /docking_controller/enable std_srvs/srv/SetBool "{data: false}"

# 삽입 상태에서 저속 배출 / 일시정지
ros2 service call /docking_controller/retract std_srvs/srv/SetBool "{data: true}"
ros2 service call /docking_controller/retract std_srvs/srv/SetBool "{data: false}"
```

## 5. 도킹 순서

```text
ACKERMANN(1) 중앙 정렬 → 완전 정지 → DIFF(2) 각도 정렬 → 진입(INSERT)
```

- `base_link +X` 로봇 전방 / `+Y` 좌측 / `-Y` 우측(현재 침대 진입 방향)
- **ACKERMANN**: `linear.x`만 사용해 침대 입구 가로판의 수직 중심선에 정렬.
  연속 3프레임 확인 후 0 속도를 5프레임 발행한다.
- **DIFF**: 실제 모드 피드백이 확인되면 `angular.z`로 제자리 회전해 yaw 정렬,
  이후 `linear.y < 0`으로 침대 안쪽에 진입한다.
- 검출은 `FOOT_RANSAC`(발판 높이층)으로 시작해 INSERT 0.20 m 이후 상판 양쪽 긴
  외곽선 기반 `TOP_OUTLINES`로 전환한다.

## 6. 주요 ROS 인터페이스

| 구분 | 이름 | 의미 |
|---|---|---|
| 입력 | `/livox_merge/merged_pointcloud` | `base_link`로 병합된 `sensor_msgs/msg/PointCloud2` |
| 입력 | `/motor_node/drive_mode` | 하부제어기 모드 상태 문자열 |
| 출력 | `/detected_dock_pose` | `base_link` 기준 침대 입구 pose |
| 출력 | `/bed_detection/confidence` | 검출 신뢰도 0~1 |
| 출력 | `/cmd_vel` | 플랫폼 속도 명령 |
| 출력 | `/mode` | 하부제어기 모드 코드 (1=ACKERMANN, 2=DIFF) |
| 출력 | `/docking_controller/status` | 제어 상태 및 정지 원인 |
| 서비스 | `/docking_controller/enable` | 도킹 시작 / 정지 |
| 서비스 | `/docking_controller/retract` | 보정 배출 시작 / 일시정지 |

전체 토픽 목록, 안전 정지 조건(`LOST`, `LOW_CONFIDENCE`, `MODE_SWITCH_TIMEOUT`,
`INSERT_ALIGNMENT_LOST` 등), 실기 점검표와 튜닝 항목은
[src/transfer_robot/README.md](src/transfer_robot/README.md)에 정리되어 있다.

## 7. 테스트

```bash
cd ~/jun_ws
source /opt/ros/humble/setup.bash
colcon test --packages-select transfer_robot
colcon test-result --verbose
```

## 8. 라이선스

**연구 목적 한정 / 상업적 이용 금지 / 저작권 표시 필수**

- 비영리 학술·연구 목적으로만 사용·복제·수정·배포할 수 있다. 연구 목적 이외의 사용은 금지된다.
- 판매, 재라이선스, 유료 서비스, 상용 제품 통합 등 일체의 상업적 이용을 금지한다.
- 재배포 시 저작권 표시와 라이선스 전문을 그대로 포함해야 하며, 연구 결과 공개 시
  원저작자를 명시해야 한다.
- 상업적 이용 등 예외는 저작권자의 사전 서면 허가가 필요하다.

전문은 [LICENSE](LICENSE)를 참조한다.

```
Copyright (c) 2026 hyeongjun (GitHub: Geppetto0608). All rights reserved.
```
