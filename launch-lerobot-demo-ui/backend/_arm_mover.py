#!/usr/bin/env python3
"""
Arm Mover Worker — moves robot arm to specified joint positions via ROS2.

Spawned as a subprocess by the backend (with ROS2 env sourced).

Usage:
  python _arm_mover.py '<json_params>'
    json_params: {"joint_names": [...], "positions": [...], "move_time": 0.7}

Output: JSON {"status": "ok"|"error", "message": "..."}
"""

import json
import sys
import time


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"status": "error", "message": "Usage: _arm_mover.py <json_params>"}))
        sys.exit(1)

    try:
        params = json.loads(sys.argv[1])
    except json.JSONDecodeError as e:
        print(json.dumps({"status": "error", "message": f"Invalid JSON: {e}"}))
        sys.exit(1)

    joint_names = params.get("joint_names", [])
    positions = params.get("positions", [])
    move_time = params.get("move_time", 0.7)

    if not joint_names or not positions:
        print(json.dumps({"status": "error", "message": "Missing joint_names or positions"}))
        sys.exit(1)

    # ── Import ROS2 packages ──
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.action import ActionClient
        from control_msgs.action import FollowJointTrajectory
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        from builtin_interfaces.msg import Duration
        from sensor_msgs.msg import JointState
        from rclpy.qos import qos_profile_sensor_data
    except ImportError as e:
        print(json.dumps({"status": "error", "message": f"ROS2 not available: {e}"}))
        sys.exit(1)

    rclpy.init()
    node = rclpy.create_node("arm_mover_worker")

    try:
        ac = ActionClient(
            node, FollowJointTrajectory, "/arm_controller/follow_joint_trajectory"
        )

        if not ac.wait_for_server(timeout_sec=5.0):
            print(json.dumps({"status": "error", "message": "Action server not available"}))
            sys.exit(1)

        # ── Get current joint positions ──
        current_state = None

        def joint_state_cb(msg):
            nonlocal current_state
            current_state = msg

        sub = node.create_subscription(
            JointState, "/joint_states", joint_state_cb, qos_profile_sensor_data
        )

        t0 = time.time()
        while current_state is None and time.time() - t0 < 2.0:
            rclpy.spin_once(node, timeout_sec=0.1)
        node.destroy_subscription(sub)

        if current_state is not None:
            cur_dict = {n: current_state.position[i] for i, n in enumerate(current_state.name)}
            start_positions = [cur_dict.get(n, 0.0) for n in joint_names]
        else:
            start_positions = positions  # fallback

        # ── Build trajectory ──
        traj = JointTrajectory()
        traj.joint_names = joint_names

        # Start point (current position)
        p0 = JointTrajectoryPoint()
        p0.positions = start_positions
        p0.velocities = [0.0] * len(joint_names)
        p0.accelerations = [0.0] * len(joint_names)
        p0.time_from_start = Duration(sec=0, nanosec=0)

        # End point (target position)
        p1 = JointTrajectoryPoint()
        p1.positions = positions
        p1.velocities = [0.0] * len(joint_names)
        p1.accelerations = [0.0] * len(joint_names)
        p1.time_from_start = Duration()
        p1.time_from_start.sec = int(move_time)
        p1.time_from_start.nanosec = int((move_time - int(move_time)) * 1e9)

        traj.points = [p0, p1]

        # ── Send goal ──
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        future = ac.send_goal_async(goal)
        rclpy.spin_until_future_complete(node, future)
        goal_handle = future.result()

        if not goal_handle.accepted:
            print(json.dumps({"status": "error", "message": "Trajectory rejected by controller"}))
            sys.exit(1)

        # Wait for execution
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(node, result_future, timeout_sec=move_time + 2.0)

        # Brief hold
        time.sleep(0.3)

        print(json.dumps({"status": "ok", "message": "Movement complete"}))

    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)[:200]}))
        sys.exit(1)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
