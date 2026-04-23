"""
VR Teleoperation Bridge for Real SO-101 Robot
==============================================
Delta-control approach (based on official xlerobot_vr.py):
  - Frame-to-frame VR position differences drive the arm (no absolute origin jump)
  - 2D IK in the arm's vertical plane with alpha=0.1 smoothing
  - Wrist flex auto-compensated to keep EE orientation level
  - On first VR position: reads current joints as target, arm stays put

Usage:
    python vr_teleop_real.py --robot_port /dev/tty.usbmodem5A7C1231311 --arm right

    # After "Combined server started" appears, run in another terminal:
    cloudflared tunnel --url http://localhost:8080

Open the cloudflared https:// URL in Quest 3, then click the VR goggles button.
"""

import argparse
import asyncio
import os
import re
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from typing import Optional

import aiohttp
from aiohttp import web

# ── Add XLeVR to path ────────────────────────────────────────────────────────
XLEVR_PATH   = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../../../XLeRobot/XLeVR"))
XLEVR_WEB_UI = os.path.join(XLEVR_PATH, "web-ui")
COMBINED_PORT = 8080

if XLEVR_PATH not in sys.path:
    sys.path.insert(0, XLEVR_PATH)

from lerobot.model.SO101Robot import SO101Kinematics, create_real_robot

try:
    from vr_monitor import VRMonitor, get_local_ip
except ImportError as e:
    print(f"[ERROR] Could not import VRMonitor: {e}")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VRTeleopConfig:
    # Single-arm mode: set robot_port + arm.  Dual-arm mode: set both *_port fields.
    robot_port: str        = "/dev/tty.usbmodem5A7C1231311"  # used when arm != "both"
    arm: str               = "right"                          # "left" | "right" | "both"
    left_robot_port: str   = "/dev/tty.usbmodem5A7C1190111"
    right_robot_port: str  = "/dev/tty.usbmodem5A7C1231311"
    robot_camera_index: int = -1
    control_hz: float = 30.0
    kp: float        = 1.0             # P-control gain (1.0 = direct, alpha smoothing handles lag)
    gripper_trigger_threshold: float = 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Read current joint angles from robot
# ─────────────────────────────────────────────────────────────────────────────

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper"]

def read_joints(robot) -> dict:
    """Return {motor_name: degrees} for all joints."""
    present = robot.bus.sync_read("Present_Position")
    return {name: float(present.get(name, 0.0)) for name in JOINT_NAMES}


# ─────────────────────────────────────────────────────────────────────────────
# Mapper — delta control (based on official xlerobot_vr.py SimpleTeleopArm)
# ─────────────────────────────────────────────────────────────────────────────

class VRToSO101Mapper:
    """
    Frame-to-frame delta control (mirrors official xlerobot_vr.py):
      - VR position delta per frame → incremental 2D IK target update
      - VR X (left/right) → shoulder_pan
      - VR Y (up/down)    → current_y (vertical reach)
      - VR Z (forward)    → current_x (horizontal reach, negated)
      - Alpha smoothing on shoulder_lift / elbow_flex prevents jerky motion
      - wrist_flex auto-compensated to keep EE orientation level
    """

    # Scaling constants (from official implementation)
    _PAN_PRE_SCALE  = 170.0   # before pos_scale
    _LIFT_PRE_SCALE = 80.0
    _Z_PRE_SCALE    = 80.0
    _POS_SCALE      = 0.015   # (units: m per pre-scaled unit)
    _ANGLE_SCALE    = 3.0     # wrist delta scale
    _DELTA_LIMIT    = 0.02    # m per frame max
    _ANGLE_LIMIT    = 6.0     # deg per frame max
    _ALPHA          = 0.1     # shoulder smoothing factor
    _PAN_XY_SCALE   = 200.0   # pan from delta_x

    def __init__(self, cfg: VRTeleopConfig, pan_sign: float = 1.0):
        self.cfg      = cfg
        self.ik       = SO101Kinematics()
        # +1 for right arm, -1 for left arm (mirrored shoulder_pan direction)
        self.pan_sign = pan_sign

        # 2D IK state (arm's vertical plane)
        self.current_x = 0.1629   # horizontal reach (m)
        self.current_y = 0.1131   # vertical height (m)
        self.pitch     = 0.0      # accumulated wrist pitch (deg)

        # P-control targets (degrees)
        self.target_positions = {j: 0.0 for j in JOINT_NAMES}

        # Previous VR values for delta computation
        self.prev_vr_pos    = None
        self.prev_wrist_flex = None
        self.prev_wrist_roll = None
        self._dbg_count     = 0

    def reset(self, robot):
        """
        Called when VR controller first sends data.
        Seeds target_positions from actual robot state so arm stays put.
        """
        try:
            joints = read_joints(robot)
        except Exception as e:
            print(f"[Mapper] WARNING: could not read joints ({e}). Using zeros.")
            joints = {k: 0.0 for k in JOINT_NAMES}

        for name in JOINT_NAMES:
            self.target_positions[name] = joints.get(name, 0.0)

        # Reset VR delta tracking
        self.prev_vr_pos     = None
        self.prev_wrist_flex = None
        self.prev_wrist_roll = None

        # Seed 2D IK state from actual robot FK so the arm doesn't drift at connect.
        MAX_REACH = self.ik.l1 + self.ik.l2 - 0.01
        try:
            fk_x, fk_y = self.ik.forward_kinematics(
                joints["shoulder_lift"], joints["elbow_flex"]
            )
            # Apply the same workspace clamp handle_vr_input would apply.
            # If the arm is outside the valid workspace (e.g. pointing backward → fk_x < 0),
            # clamp immediately and recompute lift/elbow targets from IK so there's no
            # discontinuity between target_positions and current_x/y on the first VR frame.
            cx = max(0.05, min(MAX_REACH, fk_x))
            cy = max(-0.15, min(0.20,     fk_y))
            self.current_x = cx
            self.current_y = cy

            if abs(cx - fk_x) > 1e-4 or abs(cy - fk_y) > 1e-4:
                print(f"[Mapper] WARNING: arm FK ({fk_x:.3f}, {fk_y:.3f}) outside workspace "
                      f"— clamping to ({cx:.3f}, {cy:.3f}) and re-targeting lift/elbow.")
                ik_lift, ik_elbow = self.ik.inverse_kinematics(cx, cy)
                self.target_positions["shoulder_lift"] = ik_lift
                self.target_positions["elbow_flex"]    = ik_elbow
                self.pitch = (joints["wrist_flex"] + ik_lift + ik_elbow)
            else:
                # FK is in range — targets already set from actual joints above.
                self.pitch = (joints["wrist_flex"]
                              + joints["shoulder_lift"]
                              + joints["elbow_flex"])
        except Exception as e:
            print(f"[Mapper] WARNING: FK at reset failed ({e}). Using neutral IK state.")
            self.current_x = 0.1629
            self.current_y = 0.1131
            self.pitch     = 0.0

        print("\n[Mapper] ── Reset to current robot position ─────────────")
        for name in JOINT_NAMES:
            print(f"  {name:15s} = {joints.get(name, 0.0):7.1f}°")
        print("[Mapper] ───────────────────────────────────────────────────\n")

    def handle_vr_input(self, goal):
        """Update target_positions using frame-to-frame VR deltas."""
        if goal is None or goal.target_position is None:
            return

        current_vr_pos = goal.target_position

        # First frame: just establish baseline, no movement
        if self.prev_vr_pos is None:
            self.prev_vr_pos = list(current_vr_pos)
            return

        # ── Frame-to-frame deltas, pre-scaled ────────────────────────────────
        vr_x = (current_vr_pos[0] - self.prev_vr_pos[0]) * self._PAN_PRE_SCALE
        vr_y = (current_vr_pos[1] - self.prev_vr_pos[1]) * self._LIFT_PRE_SCALE
        vr_z = (current_vr_pos[2] - self.prev_vr_pos[2]) * self._Z_PRE_SCALE
        self.prev_vr_pos = list(current_vr_pos)

        delta_x = max(-self._DELTA_LIMIT, min(self._DELTA_LIMIT, vr_x * self._POS_SCALE))
        delta_y = max(-self._DELTA_LIMIT, min(self._DELTA_LIMIT, vr_y * self._POS_SCALE))
        delta_z = max(-self._DELTA_LIMIT, min(self._DELTA_LIMIT, vr_z * self._POS_SCALE))

        # Dead zone
        thr = 0.001
        if abs(delta_x) < thr: delta_x = 0.0
        if abs(delta_y) < thr: delta_y = 0.0
        if abs(delta_z) < thr: delta_z = 0.0

        # ── Update 2D IK target ───────────────────────────────────────────────
        # VR Z (forward/back) → arm reach (negated: lean forward = reach out)
        # VR Y (up/down)      → arm height
        self.current_x += -delta_z
        self.current_y += delta_y

        # Clamp to safe workspace
        MAX_REACH = self.ik.l1 + self.ik.l2 - 0.01
        self.current_x = max(0.05,     min(MAX_REACH, self.current_x))
        self.current_y = max(-0.15,    min(0.20,      self.current_y))

        # ── Shoulder pan from VR X (side-to-side) ────────────────────────────
        # pan_sign=-1 for left arm: its shoulder_pan is physically mirrored
        if abs(delta_x) > 0.001:
            delta_pan = max(-self._ANGLE_LIMIT,
                            min(self._ANGLE_LIMIT, delta_x * self._PAN_XY_SCALE * self.pan_sign))
            self.target_positions["shoulder_pan"] = max(-90.0, min(90.0,
                self.target_positions["shoulder_pan"] + delta_pan))

        # ── 2D IK → shoulder_lift, elbow_flex with alpha smoothing ───────────
        try:
            joint2, joint3 = self.ik.inverse_kinematics(self.current_x, self.current_y)
            a = self._ALPHA
            self.target_positions["shoulder_lift"] = (
                (1 - a) * self.target_positions["shoulder_lift"] + a * joint2
            )
            self.target_positions["elbow_flex"] = (
                (1 - a) * self.target_positions["elbow_flex"] + a * joint3
            )
        except Exception as e:
            print(f"[Mapper] IK failed (x={self.current_x:.3f}, y={self.current_y:.3f}): {e}")

        # ── Wrist flex compensation (keeps EE level) ─────────────────────────
        self.target_positions["wrist_flex"] = (
            -self.target_positions["shoulder_lift"]
            - self.target_positions["elbow_flex"]
            + self.pitch
        )

        # ── Wrist roll (delta from VR rotation) ──────────────────────────────
        if hasattr(goal, 'wrist_roll_deg') and goal.wrist_roll_deg is not None:
            if self.prev_wrist_roll is None:
                self.prev_wrist_roll = goal.wrist_roll_deg
            else:
                dr = (goal.wrist_roll_deg - self.prev_wrist_roll) * self._ANGLE_SCALE
                if abs(dr) < 1.0: dr = 0.0
                dr = max(-self._ANGLE_LIMIT, min(self._ANGLE_LIMIT, dr))
                self.target_positions["wrist_roll"] = max(-90.0, min(90.0,
                    self.target_positions["wrist_roll"] + dr))
                self.prev_wrist_roll = goal.wrist_roll_deg

        # ── Wrist pitch (accumulates into self.pitch) ─────────────────────────
        if hasattr(goal, 'wrist_flex_deg') and goal.wrist_flex_deg is not None:
            if self.prev_wrist_flex is None:
                self.prev_wrist_flex = goal.wrist_flex_deg
            else:
                dp = (goal.wrist_flex_deg - self.prev_wrist_flex) * self._ANGLE_SCALE
                if abs(dp) < 1.0: dp = 0.0
                dp = max(-self._ANGLE_LIMIT, min(self._ANGLE_LIMIT, dp))
                self.pitch = max(-90.0, min(90.0, self.pitch + dp))
                self.prev_wrist_flex = goal.wrist_flex_deg

        # ── Gripper ──────────────────────────────────────────────────────────
        trigger = (goal.metadata or {}).get('trigger', 0)
        self.target_positions["gripper"] = 45.0 if trigger > self.cfg.gripper_trigger_threshold else 0.0

        # ── Debug ─────────────────────────────────────────────────────────────
        self._dbg_count += 1
        if self._dbg_count % 30 == 0:
            print(f"[DBG] VR delta  x={delta_x:+.3f}  y={delta_y:+.3f}  z={delta_z:+.3f}")
            print(f"[DBG] IK target x={self.current_x:.3f}  y={self.current_y:.3f}")
            print(f"[DBG] Joints → pan={self.target_positions['shoulder_pan']:+.1f}°"
                  f"  lift={self.target_positions['shoulder_lift']:+.1f}°"
                  f"  elbow={self.target_positions['elbow_flex']:+.1f}°"
                  f"  wrist_flex={self.target_positions['wrist_flex']:+.1f}°")

    def get_action(self, current_joints: dict) -> dict:
        """Generate P-control action from current joint readings."""
        action = {}
        kp = self.cfg.kp
        for joint in JOINT_NAMES:
            current = current_joints.get(joint, 0.0)
            target  = self.target_positions.get(joint, current)
            action[f"{joint}.pos"] = current + kp * (target - current)
        return action


# ─────────────────────────────────────────────────────────────────────────────
# Combined HTTP + WebSocket proxy (aiohttp)
# ─────────────────────────────────────────────────────────────────────────────

_cloudflare_url: str = ""   # set once tunnel is up

def make_combined_app(ws_upstream_port: int) -> web.Application:

    async def index(request):
        return web.FileResponse(os.path.join(XLEVR_WEB_UI, "index.html"))

    async def go(request):
        if _cloudflare_url:
            raise web.HTTPFound(_cloudflare_url)
        return web.Response(text="Tunnel not ready yet, try again in a moment.")

    async def ws_proxy(request):
        client_ws = web.WebSocketResponse()
        await client_ws.prepare(request)
        session = aiohttp.ClientSession()
        try:
            async with session.ws_connect(f"ws://localhost:{ws_upstream_port}") as up:

                async def c2u():
                    async for msg in client_ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            await up.send_str(msg.data)
                        elif msg.type == aiohttp.WSMsgType.BINARY:
                            await up.send_bytes(msg.data)
                        elif msg.type in (aiohttp.WSMsgType.CLOSE,
                                          aiohttp.WSMsgType.ERROR):
                            break

                async def u2c():
                    async for msg in up:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            await client_ws.send_str(msg.data)
                        elif msg.type == aiohttp.WSMsgType.BINARY:
                            await client_ws.send_bytes(msg.data)
                        elif msg.type in (aiohttp.WSMsgType.CLOSE,
                                          aiohttp.WSMsgType.ERROR):
                            break

                await asyncio.gather(c2u(), u2c(), return_exceptions=True)
        except Exception as e:
            print(f"[WS proxy] {e}")
        finally:
            await session.close()
        return client_ws

    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/go", go)
    app.router.add_get("/ws", ws_proxy)
    app.router.add_static("/", XLEVR_WEB_UI)
    return app


# ─────────────────────────────────────────────────────────────────────────────
# Cloudflared auto-start + short URL
# ─────────────────────────────────────────────────────────────────────────────

def shorten_url(long_url: str) -> str:
    """Shorten URL using services that allow cloudflare tunnel URLs."""
    apis = [
        f"https://ulvis.net/api.php?url={long_url}",
        f"https://clck.ru/--?url={long_url}",
    ]
    for api in apis:
        try:
            with urllib.request.urlopen(api, timeout=5) as r:
                short = r.read().decode().strip()
            if short.startswith("http") and len(short) < len(long_url):
                return short
        except Exception:
            continue
    return long_url

async def start_cloudflared(port: int) -> Optional[subprocess.Popen]:
    """
    Launch cloudflared tunnel, wait for the public URL, shorten it,
    and print it clearly. Returns the process so it can be terminated later.
    """
    cf_bin = "cloudflared"
    try:
        subprocess.run([cf_bin, "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("[cloudflared] Not found — install with: brew install cloudflared")
        print(f"[cloudflared] Then run manually: cloudflared tunnel --url http://localhost:{port}")
        return None

    proc = subprocess.Popen(
        [cf_bin, "tunnel", "--url", f"http://localhost:{port}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    print("[cloudflared] Starting tunnel …")
    url = None
    deadline = time.time() + 30
    while time.time() < deadline:
        line = await asyncio.get_event_loop().run_in_executor(None, proc.stdout.readline)
        if not line:
            break
        m = re.search(r"https://[a-zA-Z0-9\-]+\.trycloudflare\.com", line)
        if m:
            url = m.group(0)
            break

    if url:
        global _cloudflare_url
        _cloudflare_url = url
        local_ip   = get_local_ip()
        local_url  = f"http://{local_ip}:{port}/go"
        print()
        print("┌──────────────────────────────────────┐")
        print("│  Quest 3 browser → type this:        │")
        print(f"│  {local_url:<36s}  │")
        print("│  It will redirect to the VR page.    │")
        print("│  Then tap the VR goggles icon.       │")
        print("└──────────────────────────────────────┘")
        print()
    else:
        print("[cloudflared] Could not detect URL — check terminal output")

    return proc


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _connect_robot(port: str, camera_index: int, label: str, robot_id: str = "robot1"):
    """Helper: create, connect, and print joint angles for one arm."""
    print(f"[{label}] Connecting on {port} …")
    robot = create_real_robot(port=port, camera_index=camera_index, uid="so101", robot_id=robot_id)
    robot.connect(calibrate=True)
    try:
        joints = read_joints(robot)
        print(f"[{label}] Current joint angles:")
        for name, deg in joints.items():
            print(f"  {name:15s} = {deg:7.1f}°")
    except Exception as e:
        print(f"[{label}] Could not read joints: {e}")
    return robot


def _step_arm(arm_label: str, goal, mapper: VRToSO101Mapper,
              robot, prev_had: bool) -> bool:
    """Run one control step for one arm. Returns new prev_had_position."""
    if goal is None:
        return False

    has_position = goal.target_position is not None

    if has_position and not prev_had:
        mapper.reset(robot)

    if has_position:
        mapper.handle_vr_input(goal)
        try:
            current_joints = read_joints(robot)
            action = mapper.get_action(current_joints)
            robot.send_action(action)
        except Exception as e:
            print(f"[{arm_label}] Control error: {e}")

    return has_position


async def run(cfg: VRTeleopConfig):
    dual = (cfg.arm == "both")

    # VR monitor — WebSocket server only
    monitor = VRMonitor()
    if not monitor.initialize():
        print("[ERROR] Failed to initialize VRMonitor")
        return

    await monitor.vr_server.start()
    monitor.is_running = True

    # Combined HTTP + WS proxy
    ws_port = monitor.config.websocket_port
    runner  = web.AppRunner(make_combined_app(ws_port))
    await runner.setup()
    await web.TCPSite(runner, "0.0.0.0", COMBINED_PORT).start()

    print(f"\n🌐 Combined server on port {COMBINED_PORT}")
    cf_proc = await start_cloudflared(COMBINED_PORT)

    monitor_task = asyncio.create_task(monitor.monitor_commands())

    # Connect robot(s)
    if dual:
        robot_left  = _connect_robot(cfg.left_robot_port,  cfg.robot_camera_index, "Left",  robot_id="robot2")
        robot_right = _connect_robot(cfg.right_robot_port, cfg.robot_camera_index, "Right", robot_id="robot1")
    else:
        robot_single = _connect_robot(cfg.robot_port, cfg.robot_camera_index,
                                      cfg.arm.capitalize())

    print(f"\n[Info] mode={'dual-arm' if dual else cfg.arm}  kp={cfg.kp}  hz={cfg.control_hz}")
    print("[Info] Arms will NOT move until Quest 3 VR session starts.")
    print("[Info] Press Ctrl+C to stop.\n")

    dt = 1.0 / cfg.control_hz

    if dual:
        mapper_left  = VRToSO101Mapper(cfg, pan_sign=+1.0)
        mapper_right = VRToSO101Mapper(cfg, pan_sign=+1.0)
        prev_left  = False
        prev_right = False
    else:
        pan_sign = +1.0 if cfg.arm == "left" else 1.0
        mapper_single = VRToSO101Mapper(cfg, pan_sign=pan_sign)
        prev_single   = False

    try:
        while True:
            t_start = time.perf_counter()

            if dual:
                goal_left  = monitor.get_latest_goal_nowait("left")
                goal_right = monitor.get_latest_goal_nowait("right")
                prev_left  = _step_arm("Left",  goal_left,  mapper_left,  robot_left,  prev_left)
                prev_right = _step_arm("Right", goal_right, mapper_right, robot_right, prev_right)
            else:
                goal = monitor.get_latest_goal_nowait(cfg.arm)
                prev_single = _step_arm(cfg.arm.capitalize(), goal,
                                        mapper_single, robot_single, prev_single)

            elapsed = time.perf_counter() - t_start
            await asyncio.sleep(max(0.0, dt - elapsed))

    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n[Info] Shutting down …")
    finally:
        monitor_task.cancel()
        await monitor.vr_server.stop()
        await runner.cleanup()
        if dual:
            for label, robot in [("Left", robot_left), ("Right", robot_right)]:
                try:
                    robot.disconnect()
                except Exception as e:
                    print(f"[{label}] Disconnect error (motor may be in overload): {e}")
        else:
            try:
                robot_single.disconnect()
            except Exception as e:
                print(f"[{cfg.arm.capitalize()}] Disconnect error (motor may be in overload): {e}")
        if cf_proc:
            cf_proc.terminate()
        print("[Robot] Disconnected.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="VR teleop: Meta Quest 3 → SO-101 (single or dual arm)")
    p.add_argument("--arm",              default="both", choices=["left", "right", "both"],
                   help="Which arm to control (default: both)")
    p.add_argument("--robot_port",       default="/dev/tty.usbmodem5A7C1231311",
                   help="Port for single-arm mode (--arm left or right)")
    p.add_argument("--left_robot_port",  default="/dev/tty.usbmodem5A7C1190111",
                   help="Left arm port (used when --arm both)")
    p.add_argument("--right_robot_port", default="/dev/tty.usbmodem5A7C1231311",
                   help="Right arm port (used when --arm both)")
    p.add_argument("--control_hz",       type=float, default=30.0)
    p.add_argument("--kp",               type=float, default=1.0,
                   help="P-control gain (1.0=direct, lower=smoother but laggier)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg  = VRTeleopConfig(
        arm=args.arm,
        robot_port=args.robot_port,
        left_robot_port=args.left_robot_port,
        right_robot_port=args.right_robot_port,
        control_hz=args.control_hz,
        kp=args.kp,
    )
    asyncio.run(run(cfg))
