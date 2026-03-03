#!/usr/bin/env python3
"""
Move SO-ARM robot through joint positions from CSV file — AUTO SEQUENTIAL MODE
自动按顺序执行所有位置点，每一步的 move_time 和 hold_time 可单独调整。
"""
import csv
import sys
import time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration


# ════════════════════════════════════════════════════════════════════════
#  配置区 — 在这里调整每一步的时间
# ════════════════════════════════════════════════════════════════════════

CSV_FILE = "cloth.csv"

# 每一步的时间配置: (move_time, hold_time)
#   move_time = 到达该位置的运动时间（秒）
#   hold_time = 到达后停留时间（秒）
#
# 数组长度应与 CSV 中的点数一致。
# 如果数组比点数短，多出的点使用 DEFAULT_TIMING。
# 如果数组比点数长，多出的配置会被忽略。

DEFAULT_TIMING = (0.7, 0.7)  # 默认: move 0.7s, hold 0.7s

STEP_TIMINGS = [
    # (move_time, hold_time)
    (0.4, 0.0),   # Step 1
    (1.2, 0.0),   # Step 2
    (0.4, 0.0),   # Step 3
    (1.2, 0.0),   # Step 4
    (0.4, 0.0),   # Step 5
    (1.2, 0.0),   # Step 6
    (0.7, 0.0),   # Step 7
    (0.7, 0.0),   # Step 8
    (0.5, 0.8),   # Step 9
    (0.5, 0.8),   # Step 10
]

# ════════════════════════════════════════════════════════════════════════


class ArmController(Node):
    def __init__(self):
        super().__init__("arm_controller_node")
        self.action_client = ActionClient(
            self, FollowJointTrajectory, "/arm_controller/follow_joint_trajectory"
        )
        self.get_logger().info("Arm controller node initialized")

    def get_current_joint_positions(self):
        """Get current joint positions from /joint_states"""
        from sensor_msgs.msg import JointState
        from rclpy.qos import qos_profile_sensor_data

        sub = self.create_subscription(
            JointState,
            "/joint_states",
            lambda msg: setattr(self, '_current_joint_state', msg),
            qos_profile_sensor_data
        )

        self._current_joint_state = None
        timeout = time.time() + 2.0
        while self._current_joint_state is None and time.time() < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)

        self.destroy_subscription(sub)

        if self._current_joint_state is None:
            return None

        current_positions = {}
        for i, name in enumerate(self._current_joint_state.name):
            current_positions[name] = self._current_joint_state.position[i]

        return current_positions

    def move_to_joint_positions(self, joint_positions, joint_names, move_time=0.7, hold_time=0.7):
        """Move robot to specified joint positions."""
        if not self.action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("arm_controller action server not available")
            return False

        current_positions = self.get_current_joint_positions()
        if current_positions is None:
            self.get_logger().warn("Could not get current joint positions, using target as start")
            start_positions = joint_positions
        else:
            start_positions = [current_positions.get(name, 0.0) for name in joint_names]

        traj = JointTrajectory()
        traj.joint_names = joint_names

        start_point = JointTrajectoryPoint()
        start_point.positions = start_positions
        start_point.velocities = [0.0] * len(joint_names)
        start_point.accelerations = [0.0] * len(joint_names)
        start_point.time_from_start = Duration(sec=0, nanosec=0)

        end_point = JointTrajectoryPoint()
        end_point.positions = joint_positions
        end_point.velocities = [0.0] * len(joint_names)
        end_point.accelerations = [0.0] * len(joint_names)
        end_point.time_from_start = Duration()
        end_point.time_from_start.sec = int(move_time)
        end_point.time_from_start.nanosec = int((move_time - int(move_time)) * 1e9)

        traj.points = [start_point, end_point]

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        self.get_logger().info(
            f"Moving from {[f'{p:.4f}' for p in start_positions]} "
            f"to {[f'{p:.4f}' for p in joint_positions]}"
        )

        future = self.action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error("Trajectory rejected")
            return False

        self.get_logger().info("Trajectory accepted, executing...")

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=move_time + 1.0)

        time.sleep(hold_time)

        return True


def read_csv_points(csv_file):
    """Read joint positions from CSV file"""
    points = []
    joint_names = None

    try:
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i == 0:
                    joint_names = row
                    continue
                if not row or all(not cell.strip() for cell in row):
                    continue
                positions = [float(x) for x in row]
                points.append(positions)
    except FileNotFoundError:
        print(f"Error: File {csv_file} not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    return points, joint_names


def main():
    # Read points from CSV
    print(f"Reading joint positions from {CSV_FILE}...")
    points, joint_names = read_csv_points(CSV_FILE)

    if not points:
        print("Error: No points found in CSV file")
        sys.exit(1)

    if not joint_names:
        joint_names = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]

    total = len(points)
    print(f"Found {total} points")
    print(f"Joint names: {joint_names}")
    print()

    # Show timing plan
    print("═" * 60)
    print("  Execution Plan")
    print("═" * 60)
    for i in range(total):
        mt, ht = STEP_TIMINGS[i] if i < len(STEP_TIMINGS) else DEFAULT_TIMING
        print(f"  Step {i+1:2d}:  move {mt:.2f}s  hold {ht:.2f}s")
    print("═" * 60)
    print()

    # Initialize ROS2
    rclpy.init()
    controller = ArmController()

    try:
        print("Waiting for action server...")
        time.sleep(1.0)
        print("✅ Ready! Starting auto sequence...\n")

        for i, point in enumerate(points):
            mt, ht = STEP_TIMINGS[i] if i < len(STEP_TIMINGS) else DEFAULT_TIMING

            print(f"── Step {i+1}/{total}  (move={mt:.2f}s, hold={ht:.2f}s) ──")
            success = controller.move_to_joint_positions(point, joint_names, mt, ht)

            if success:
                print(f"  ✅ Step {i+1} done")
            else:
                print(f"  ❌ Step {i+1} failed")
            print()

        print("🎉 All steps completed!")

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
