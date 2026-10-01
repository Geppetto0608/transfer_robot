"""Small numeric terminal menu for the docking controller services."""

from __future__ import annotations

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import SetBool


def stop_service_for_phase(phase: str) -> str:
    """Select a stop service without changing steering mode unnecessarily."""
    return "retract" if phase.strip().upper() == "RETRACT" else "enable"


class DockingConsole(Node):
    def __init__(self) -> None:
        super().__init__("docking_console")
        self.enable_client = self.create_client(
            SetBool, "/docking_controller/enable"
        )
        self.retract_client = self.create_client(
            SetBool, "/docking_controller/retract"
        )
        self.status = "UNKNOWN"
        self.phase = "UNKNOWN"
        self.create_subscription(
            String, "/docking_controller/status", self.status_callback, 10
        )
        self.create_subscription(
            String, "/docking_phase", self.phase_callback, 10
        )

    def status_callback(self, message: String) -> None:
        self.status = message.data

    def phase_callback(self, message: String) -> None:
        self.phase = message.data

    def refresh_state(self) -> None:
        # The controller publishes both topics continuously at 20 Hz.
        for _ in range(3):
            rclpy.spin_once(self, timeout_sec=0.05)

    def print_state(self) -> None:
        self.refresh_state()
        print(f"현재 상태: {self.status} / phase: {self.phase}")

    def call(self, service: str, value: bool) -> None:
        client = self.enable_client if service == "enable" else self.retract_client
        if not client.wait_for_service(timeout_sec=2.0):
            print("서비스가 없습니다. transfer_robot.launch.py를 먼저 실행하세요.")
            return
        request = SetBool.Request()
        request.data = value
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if not future.done() or future.result() is None:
            print("서비스 응답 시간 초과")
            return
        response = future.result()
        result = "성공" if response.success else "실패"
        print(f"{result}: {response.message}")
        self.print_state()

    def stop_current_motion(self) -> None:
        self.refresh_state()
        service = stop_service_for_phase(self.phase)
        self.call(service, False)


MENU = """
========== BED DOCKING ==========
1 : 진입 시작
2 : 배출 시작 (3번 누를 때까지 계속)
3 : 현재 동작 정지
4 : ACKERMANN 복귀 / 전체 해제
5 : 상태 확인
q : 콘솔 종료
=================================
"""


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DockingConsole()
    try:
        print(MENU)
        node.print_state()
        while rclpy.ok():
            try:
                choice = input("선택> ").strip().lower()
            except EOFError:
                break
            if choice == "1":
                node.call("enable", True)
            elif choice == "2":
                node.call("retract", True)
            elif choice == "3":
                node.stop_current_motion()
            elif choice == "4":
                node.call("enable", False)
            elif choice == "5":
                node.print_state()
            elif choice in ("q", "quit", "exit"):
                break
            else:
                print("1, 2, 3, 4, 5 또는 q를 입력하세요.")
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
