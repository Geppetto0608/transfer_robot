# Patient Transfer Robot — ROS 2 Humble

이 워크스페이스의 메인 패키지는 `src/transfer_robot`이다. Isaac Sim, USD 모델,
시뮬레이터 드라이버와 시뮬레이션 launch는 제거했으며 실제 4WS 플랫폼만 대상으로 한다.

## 빠른 시작

```bash
cd /home/test/jun_ws
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select transfer_robot --symlink-install
source install/setup.bash
ros2 launch transfer_robot transfer_robot.launch.py
```

제어기는 항상 비활성 상태로 시작한다. LiDAR, TF, 모터 모드 피드백과 비상 정지 장치를
확인한 뒤에만 다음 서비스로 도킹을 시작한다.

도킹 순서는 `ACKERMANN(1) 중앙 정렬 → 완전 정지 → DIFF(2) 각도 정렬 → 진입`이다.
침대가 로봇 우측에 있다는 현재 TF 기준으로 ACKERMANN은 `base_link X` 방향으로
침대 판 중앙에 맞춘다. DIFF는 제자리 회전으로 yaw를 맞춘 뒤 `linear.y < 0`으로
`base_link -Y` 방향에 진입한다.

```bash
ros2 service call /docking_controller/enable std_srvs/srv/SetBool "{data: true}"
```

정지 명령:

```bash
ros2 service call /docking_controller/enable std_srvs/srv/SetBool "{data: false}"
```

토픽, 좌표계, 실기 점검표와 튜닝 항목은 `src/transfer_robot/README.md`에 정리되어 있다.
