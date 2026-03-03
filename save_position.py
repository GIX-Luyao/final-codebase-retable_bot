#!/usr/bin/env python3
"""Save current robot arm position to a JSON file.

Usage:
    python save_position.py <file> [name]
    python save_position.py <file> --clear

    <file>   — 目标文件，支持以下简写或完整路径:
               cup / lemon / tissue / cloth / item
               也可以直接写完整路径，如 /home/robotlab/lerobot/my_positions.json

    [name]   — 可选，给这个位置起个名字
    --clear  — 清空该文件中所有已保存的位置

Examples:
    python save_position.py cup           # 保存到 cup.json
    python save_position.py lemon p1      # 保存到 lemon.json，命名为 p1
    python save_position.py tissue        # 保存到 tissue.json
    python save_position.py cloth grab    # 保存到 cloth.json，命名为 grab
    python save_position.py cup --clear   # 清空 cup.json 中所有位置
"""

import json
import os
import sys
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

ROBOT_PORT = "/dev/ttyACM0"
ROBOT_ID = "follower_hope"
BASE_DIR = "/home/robotlab/lerobot"

# 简写映射 — 可以随时添加更多
FILE_ALIASES = {
    "cup":    os.path.join(BASE_DIR, "cup.json"),
    "lemon":  os.path.join(BASE_DIR, "lemon.json"),
    "tissue": os.path.join(BASE_DIR, "tissue.json"),
    "cloth":  os.path.join(BASE_DIR, "cloth.json"),
    "item":   os.path.join(BASE_DIR, "item.json"),
}

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def resolve_file(arg):
    """将简写或路径解析为完整的 JSON 文件路径。"""
    if arg in FILE_ALIASES:
        return FILE_ALIASES[arg]
    # 如果不是简写，当作路径处理
    path = os.path.abspath(arg)
    if not path.endswith(".json"):
        path += ".json"
    return path


def load_or_create(json_path):
    """加载已有 JSON，不存在则自动创建。"""
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            return json.load(f)
    else:
        print(f"📄 文件不存在，自动创建: {json_path}")
        return {"positions": []}


def main():
    if len(sys.argv) < 2:
        print("用法: python save_position.py <file> [name]")
        print()
        print("可用的文件简写:")
        for alias, path in FILE_ALIASES.items():
            exists = "✓" if os.path.exists(path) else "✗"
            print(f"  {exists} {alias:10s} → {path}")
        sys.exit(1)

    file_arg = sys.argv[1]
    json_path = resolve_file(file_arg)

    # Check for --clear flag
    if "--clear" in sys.argv[2:]:
        short_name = os.path.basename(json_path)
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                data = json.load(f)
            count = len(data.get("positions", []))
            with open(json_path, "w") as f:
                json.dump({"positions": []}, f, indent=2)
            print(f"🗑️  已清空 {short_name} (删除了 {count} 个位置)")
        else:
            print(f"⚠️  文件不存在: {short_name}，无需清空")
        sys.exit(0)

    name = sys.argv[2] if len(sys.argv) > 2 else None

    # Connect to robot
    print(f"🤖 连接机械臂...")
    robot_cfg = SO101FollowerConfig(port=ROBOT_PORT, id=ROBOT_ID, cameras={})
    robot = SO101Follower(robot_cfg)
    robot.connect()

    # Read current position
    obs = robot.get_observation()
    position = {}
    for jname in JOINT_NAMES:
        val = obs[f"{jname}.pos"]
        # 转为 Python float（避免 numpy/tensor 序列化问题）
        position[jname] = float(val) if hasattr(val, 'item') else float(val)

    robot.disconnect()

    # Load or create JSON
    data = load_or_create(json_path)

    # Add new position
    idx = len(data["positions"]) + 1
    entry = {
        "index": idx,
        "values": position,
    }
    if name:
        entry["name"] = name

    data["positions"].append(entry)

    # Save
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    label = f"'{name}'" if name else f"#{idx}"
    short_name = os.path.basename(json_path)
    print(f"✓ 位置 {label} 已保存到 {short_name} (共 {idx} 个点)")
    print(f"  {position}")


if __name__ == "__main__":
    main()
