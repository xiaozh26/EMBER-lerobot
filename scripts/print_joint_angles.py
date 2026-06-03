#!/usr/bin/env python
"""
Print joint angles of both arms in real time.

Uses the same robot IDs as vr_teleop_real.py so calibration files load automatically:
  left  arm → robot_id="robot2"  (~/.cache/.../so_follower/robot2.json)
  right arm → robot_id="robot1"  (~/.cache/.../so_follower/robot1.json)

Example:
    python print_joint_angles.py \
      --left_port  /dev/tty.usbmodemXXXX \
      --right_port /dev/tty.usbmodemXXXX \
      --fps 10
"""

import argparse
import time

from model.SO101Robot import create_real_robot
from lerobot.utils.robot_utils import precise_sleep

MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def parse_args():
    p = argparse.ArgumentParser(description="Print bimanual robot joint angles in real time.")
    p.add_argument("--left_port",  default="/dev/tty.usbmodemXXXX", help="Serial port for the left arm.")
    p.add_argument("--left_id",    default="robot2", help="Calibration ID for the left arm (default: robot2).")
    p.add_argument("--right_port", default="/dev/tty.usbmodemXXXX", help="Serial port for the right arm.")
    p.add_argument("--right_id",   default="robot1", help="Calibration ID for the right arm (default: robot1).")
    p.add_argument("--fps",        type=int, default=10, help="Print frequency in Hz (default: 10).")
    return p.parse_args()


def read_joints(robot) -> dict:
    present = robot.bus.sync_read("Present_Position")
    return {name: float(present.get(name, 0.0)) for name in MOTOR_NAMES}


def print_joints(left: dict, right: dict, step: int) -> None:
    print(f"\n--- step {step} ---")
    print(f"{'Joint':<20} {'Left (deg)':>12} {'Right (deg)':>13}")
    print("-" * 47)
    for motor in MOTOR_NAMES:
        print(f"{motor:<20} {left.get(motor, float('nan')):>12.2f} {right.get(motor, float('nan')):>13.2f}")


def main():
    args = parse_args()

    print(f"Connecting left  arm (port={args.left_port},  id={args.left_id})...")
    left_arm = create_real_robot(port=args.left_port, camera_index=-1, uid="so101", robot_id=args.left_id)
    left_arm.connect(calibrate=True)

    print(f"Connecting right arm (port={args.right_port}, id={args.right_id})...")
    right_arm = create_real_robot(port=args.right_port, camera_index=-1, uid="so101", robot_id=args.right_id)
    right_arm.connect(calibrate=True)

    print("Connected. Press Ctrl+C to stop.\n")

    dt = 1.0 / args.fps
    step = 0
    try:
        while True:
            t0 = time.perf_counter()
            left_joints  = read_joints(left_arm)
            right_joints = read_joints(right_arm)
            print_joints(left_joints, right_joints, step)
            step += 1
            precise_sleep(max(dt - (time.perf_counter() - t0), 0.0))
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        left_arm.disconnect()
        right_arm.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
