"""
VR Teleoperation Bridge for Real SO-101 Robot
==============================================
Origin-relative control (based on official xlerobot_vr.py):
  - 2D IK in the arm's vertical plane with alpha=0.1 smoothing
  - Wrist flex auto-compensated to keep EE orientation level
  - GRIP CLUTCH: the arm only moves while you squeeze GRIP. Grip press anchors at
    the arm's current physical pose (reads the motors) and the hand then moves it
    relative to that; releasing grip freezes the arm so you can reposition your
    hand and grip again. The robot never moves unless you are actively gripping.

Usage:
    python vr_teleop_real.py --robot_port /dev/tty.usbmodemXXXX --arm right

    # After "Combined server started" appears, run in another terminal:
    cloudflared tunnel --url http://localhost:8080

On startup the arm smoothly homes to the SAME fixed per-arm home pose the sim loads
(HOME_POSE), so every run begins from an identical (folded/parked) pose; squeezing
grip then unfolds it into the teleop workspace (use --no-home to skip the startup move).

A viser scene mirrors the real robot's ACTUAL joint angles in real time at
http://localhost:8081 — open it on this computer to watch the robot (--no-viz to skip).

DOUBLE-CLICK grip (two quick presses) → the arms return to the home pose and the
session starts over, exactly like quitting and rerunning. On Ctrl+C the arms also
return to the home pose before disconnecting.

Open the cloudflared https:// URL in Quest 3, tap the VR goggles button, then
SQUEEZE GRIP to start controlling (trigger = gripper).

Note: the grip clutch needs gripActive forwarded in the goal metadata — a one-line
additive edit in XLeRobot/XLeVR/xlevr/inputs/vr_ws_server.py. Without it, grip reads
False and the arm won't move (fail-safe).
"""

import argparse
import asyncio
import math
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

# ── Optional viser mirror (visualize the real robot's actual joints) ──────────
try:
    import numpy as np
    import viser
    import viser.extras
    import yourdfpy
    _VIZ_AVAILABLE = True
except ImportError:
    _VIZ_AVAILABLE = False

# ── Add XLeVR to path ────────────────────────────────────────────────────────
XLEVR_PATH   = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../../../XLeRobot/XLeVR"))
XLEVR_WEB_UI = os.path.join(XLEVR_PATH, "web-ui")
URDF_PATH    = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "../../../../XLeRobot/simulation/Maniskill/assets/xlerobot/xlerobot.urdf"))
COMBINED_PORT = 8080   # VR web page + WS proxy (cloudflared target)
VIZ_PORT      = 8081   # viser mirror of the real robot (open on this computer)

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
    robot_port: str        = "/dev/tty.usbmodemXXXX"  # used when arm != "both"
    arm: str               = "right"                          # "left" | "right" | "both"
    left_robot_port: str   = "/dev/tty.usbmodemXXXX"
    right_robot_port: str  = "/dev/tty.usbmodemXXXX"
    robot_camera_index: int = -1
    control_hz: float = 30.0
    kp: float        = 1.0             # P-control gain (1.0 = direct, alpha smoothing handles lag)
    gripper_trigger_threshold: float = 0.5
    home_on_start: bool = True         # ramp the arm to the fixed sim rest pose at startup
                                       # so every run begins from the same pose (--no-home to skip)
    home_seconds: float = 2.0          # duration of the smooth homing ramp
    viz: bool = True                   # mirror the real robot's actual joints in a viser scene
                                       # at http://localhost:8081 (--no-viz to skip)


# ─────────────────────────────────────────────────────────────────────────────
# Read current joint angles from robot
# ─────────────────────────────────────────────────────────────────────────────

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper"]

def read_joints(robot) -> dict:
    """Return {motor_name: degrees} for all joints."""
    present = robot.bus.sync_read("Present_Position")
    return {name: float(present.get(name, 0.0)) for name in JOINT_NAMES}


def headset_yaw_deg(headset_goal) -> Optional[float]:
    """Extract the headset yaw (world rotation about the up axis, degrees).

    XLeVR packs the headset euler-Y (yaw) into the goal's wrist_roll_deg field
    (see vr_ws_server.py). Returns None when no headset data is available, in
    which case the mapper falls back to world-frame (session-start) control.
    """
    if headset_goal is None:
        return None
    return getattr(headset_goal, "wrist_roll_deg", None)


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

    # ── Absolute (origin-relative) control ──────────────────────────────────
    # The hand pose at grip is the origin; every frame the arm target is
    # recomputed from the hand's displacement-from-origin (not accumulated), so
    # a given hand pose always maps to the same arm pose. It returns exactly —
    # even after saturating a workspace limit, since nothing is integrated.
    # Alpha smoothing on the outputs rejects VR tracking glitches. Gains are
    # chosen to match the previous delta feel (= old pre_scale * pos_scale).
    _REACH_GAIN   = 1.2     # m reach  per m of hand-forward displacement
    _HEIGHT_GAIN  = 1.2     # m height per m of hand-up      displacement
    _PAN_GAIN     = 510.0   # deg pan  per m of hand-right   displacement
    _ANGLE_SCALE  = 3.0     # deg wrist per deg of controller tilt (pitch & roll)
    _ROLL_DELTA_LIMIT = 90.0 # max wrist-roll change from the gripped baseline
    _ALPHA        = 0.1     # smoothing factor for lift / elbow / pan
    _HEADSET_YAW_SIGN = 1.0 # flip to -1.0 if forward/back & left/right are swapped

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

        # Origin captured on the first VR frame after each reset/grip
        self.origin_vr_pos         = None   # hand position at grip (m)
        self.origin_yaw            = None   # headset yaw at grip (deg)
        self.origin_wrist_flex_deg = None   # controller pitch at grip (deg)
        self.origin_wrist_roll_deg = None   # controller roll  at grip (deg)
        # Arm pose at grip — displacement is mapped on top of these baselines
        self.base_x     = self.current_x
        self.base_y     = self.current_y
        self.base_pan   = 0.0
        self.base_pitch = 0.0
        self.base_roll  = 0.0
        self._dbg_count = 0

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

        # Recapture origin on the next VR frame so the arm holds its current
        # pose until the hand actually moves away from the new origin.
        self.origin_vr_pos         = None
        self.origin_yaw            = None
        self.origin_wrist_flex_deg = None
        self.origin_wrist_roll_deg = None

        # Seed 2D IK state from the actual robot pose.
        MAX_REACH = self.ik.l1 + self.ik.l2 - 0.01
        try:
            fk_x, fk_y = self.ik.forward_kinematics(
                joints["shoulder_lift"], joints["elbow_flex"]
            )
            in_workspace = (0.05 <= fk_x <= MAX_REACH) and (-0.15 <= fk_y <= 0.20)
            if in_workspace:
                # Reachable pose — anchor exactly here so the arm holds still and
                # then moves relative to your hand (no drift).
                self.current_x = fk_x
                self.current_y = fk_y
            else:
                # Parked OUTSIDE the reach workspace (e.g. the folded HOME_POSE,
                # elbow≈97° beyond the reach-IK range). Keep target_positions at the
                # actual (folded) joints and seed the ready reach, so the first VR
                # frames UNFOLD the arm smoothly (alpha-smoothed) toward the ready
                # rest instead of snapping/racing the servos.
                print(f"[Mapper] arm parked outside reach workspace (FK {fk_x:.3f}, {fk_y:.3f}) "
                      f"— will unfold to the ready reach on grip.")
                self.current_x = 0.1629
                self.current_y = 0.1131
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

    def handle_vr_input(self, goal, headset_yaw_deg=None):
        """Update target_positions from the hand pose using ORIGIN-RELATIVE control.

        The hand pose at grip is captured as an origin. Every frame the arm
        target is recomputed from the hand's *displacement from that origin*
        (never accumulated), so a given hand pose always maps to the same arm
        pose — returning the hand returns the arm exactly, even after touching a
        workspace limit. Outputs are alpha-smoothed so a VR tracking glitch
        ramps in rather than snapping the arm.

        headset_yaw_deg (captured at grip): defines "forward" so the mapping is
        body-relative regardless of how you stand. None → session-start world frame.
        """
        if goal is None or goal.target_position is None:
            return

        current_vr_pos = goal.target_position

        # ── Capture origin on the first frame after a reset/grip ─────────────
        if self.origin_vr_pos is None:
            self.origin_vr_pos = list(current_vr_pos)
            self.origin_yaw    = headset_yaw_deg
            self.base_x     = self.current_x
            self.base_y     = self.current_y
            self.base_pan   = self.target_positions["shoulder_pan"]
            self.base_pitch = self.pitch
            self.base_roll  = self.target_positions["wrist_roll"]
            self.origin_wrist_flex_deg = getattr(goal, "wrist_flex_deg", None)
            self.origin_wrist_roll_deg = getattr(goal, "wrist_roll_deg", None)
            return

        # ── Hand displacement from the origin (VR world frame) ───────────────
        disp_x = current_vr_pos[0] - self.origin_vr_pos[0]
        disp_y = current_vr_pos[1] - self.origin_vr_pos[1]
        disp_z = current_vr_pos[2] - self.origin_vr_pos[2]

        # ── Body-relative remap: rotate horizontal displacement by grip yaw ──
        # forward = −Z, right = +X when yaw = 0 (session-start world frame).
        if self.origin_yaw is not None:
            th = math.radians(self.origin_yaw) * self._HEADSET_YAW_SIGN
            s, c = math.sin(th), math.cos(th)
            forward = -(disp_x * s + disp_z * c)
            right   =  (disp_x * c - disp_z * s)
        else:
            forward = -disp_z
            right   =  disp_x

        # ── Absolute reach / height targets (recomputed, never accumulated) ──
        MAX_REACH = self.ik.l1 + self.ik.l2 - 0.01
        self.current_x = max(0.05,  min(MAX_REACH, self.base_x + forward * self._REACH_GAIN))
        self.current_y = max(-0.15, min(0.20,      self.base_y + disp_y  * self._HEIGHT_GAIN))

        a = self._ALPHA

        # ── Absolute shoulder pan, alpha-smoothed (pan_sign mirrors left arm) ─
        pan_target = max(-90.0, min(90.0,
            self.base_pan + right * self._PAN_GAIN * self.pan_sign))
        self.target_positions["shoulder_pan"] = (
            (1 - a) * self.target_positions["shoulder_pan"] + a * pan_target
        )

        # ── 2D IK → shoulder_lift, elbow_flex with alpha smoothing ───────────
        try:
            joint2, joint3 = self.ik.inverse_kinematics(self.current_x, self.current_y)
            self.target_positions["shoulder_lift"] = (
                (1 - a) * self.target_positions["shoulder_lift"] + a * joint2
            )
            self.target_positions["elbow_flex"] = (
                (1 - a) * self.target_positions["elbow_flex"] + a * joint3
            )
        except Exception as e:
            print(f"[Mapper] IK failed (x={self.current_x:.3f}, y={self.current_y:.3f}): {e}")

        # ── Absolute wrist pitch (controller tilt vs. grip tilt) ─────────────
        if getattr(goal, "wrist_flex_deg", None) is not None:
            if self.origin_wrist_flex_deg is None:
                self.origin_wrist_flex_deg = goal.wrist_flex_deg
            self.pitch = max(-90.0, min(90.0,
                self.base_pitch
                + (goal.wrist_flex_deg - self.origin_wrist_flex_deg) * self._ANGLE_SCALE))

        # ── Wrist flex = level compensation (keeps EE level) + intentional pitch
        self.target_positions["wrist_flex"] = (
            -self.target_positions["shoulder_lift"]
            - self.target_positions["elbow_flex"]
            + self.pitch
        )

        # ── Absolute wrist roll (controller roll vs. grip roll) ──────────────
        if getattr(goal, "wrist_roll_deg", None) is not None:
            if self.origin_wrist_roll_deg is None:
                self.origin_wrist_roll_deg = goal.wrist_roll_deg
            roll_delta = (goal.wrist_roll_deg - self.origin_wrist_roll_deg) * self._ANGLE_SCALE
            roll_delta = max(-self._ROLL_DELTA_LIMIT, min(self._ROLL_DELTA_LIMIT, roll_delta))
            self.target_positions["wrist_roll"] = self.base_roll + roll_delta

        # ── Gripper ──────────────────────────────────────────────────────────
        trigger = (goal.metadata or {}).get('trigger', 0)
        self.target_positions["gripper"] = 45.0 if trigger > self.cfg.gripper_trigger_threshold else 0.0

        # ── Debug ─────────────────────────────────────────────────────────────
        self._dbg_count += 1
        if self._dbg_count % 30 == 0:
            yaw_str = f"{self.origin_yaw:+.0f}°" if self.origin_yaw is not None else "world"
            print(f"[DBG] disp   fwd={forward:+.4f}  right={right:+.4f}  up={disp_y:+.4f}  gripYaw={yaw_str}")
            print(f"[DBG] target x={self.current_x:.3f}  y={self.current_y:.3f}")
            print(f"[DBG] joints → pan={self.target_positions['shoulder_pan']:+.1f}°"
                  f"  lift={self.target_positions['shoulder_lift']:+.1f}°"
                  f"  elbow={self.target_positions['elbow_flex']:+.1f}°"
                  f"  wrist_flex={self.target_positions['wrist_flex']:+.1f}°"
                  f"  wrist_roll={self.target_positions['wrist_roll']:+.1f}°")

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


# Home / parking pose (servo degrees), per arm — measured on the real arms. The arms
# ramp here on startup, shutdown, and double-click restart, and the sim loads the same
# pose. It is a folded/parked configuration (shoulder_lift≈-99, elbow_flex≈97); that
# elbow is beyond the reach-IK's range, so squeezing grip UNFOLDS the arm into the
# teleop workspace and then tracks your hand. Keep vr_teleop_sim.py's HOME_POSE in sync.
HOME_POSE = {
    "left":  {"shoulder_pan": -7.60, "shoulder_lift": -99.47, "elbow_flex": 97.98,
              "wrist_flex": -2.11, "wrist_roll": -14.64, "gripper": 0.0},
    "right": {"shoulder_pan": -3.30, "shoulder_lift": -99.03, "elbow_flex": 96.35,
              "wrist_flex":  2.73, "wrist_roll": -178.07, "gripper": 0.0},
}

def home_pose(side: str) -> dict:
    """Fixed home/parking pose (servo degrees) for an arm side ('left'/'right')."""
    return dict(HOME_POSE.get(side, HOME_POSE["right"]))


# ─────────────────────────────────────────────────────────────────────────────
# Viser mirror — render the real robot's ACTUAL joints in the xlerobot URDF
# (same servo→URDF radian conversion as vr_teleop_sim.py, verified vs ManiSkill IK)
# ─────────────────────────────────────────────────────────────────────────────

_PAN_SIGN_URDF  = 1.0    # flip if shoulder_pan looks mirrored in viser
_ROLL_SIGN_URDF = 1.0    # flip if wrist_roll looks mirrored in viser
_N_DOF = 17
# joint name → (right-arm dof index, left-arm dof index) in yourdfpy's cfg array
_ARM_DOF = {
    "shoulder_pan":  (3, 9),
    "shoulder_lift": (4, 10),
    "elbow_flex":    (5, 11),
    "wrist_flex":    (6, 12),
    "wrist_roll":    (7, 13),
    "gripper":       (8, 14),
}

def _servo_deg_to_urdf_rad(servo: dict) -> dict:
    """SO-101 servo degrees → raw xlerobot URDF radians."""
    return {
        "shoulder_pan":  _PAN_SIGN_URDF  * math.radians(servo["shoulder_pan"]),
        "shoulder_lift": math.radians(90.0 - servo["shoulder_lift"]),
        "elbow_flex":    math.radians(servo["elbow_flex"] + 90.0),
        "wrist_flex":    math.radians(servo["wrist_flex"]),
        "wrist_roll":    _ROLL_SIGN_URDF * math.radians(servo["wrist_roll"]),
        "gripper":       math.radians(servo["gripper"]),
    }

def _build_view_cfg(arm_joints: dict):
    """arm_joints: {'left': {servo deg…}, 'right': {…}} for the connected arm(s).
    Any arm not present renders at its home pose so the robot looks natural."""
    cfg = np.zeros(_N_DOF)
    for side in ("left", "right"):
        urdf = _servo_deg_to_urdf_rad(arm_joints.get(side, home_pose(side)))
        side_idx = 1 if side == "left" else 0
        for j, (r_idx, l_idx) in _ARM_DOF.items():
            cfg[l_idx if side_idx else r_idx] = urdf[j]
    return cfg


async def _home_arms(robots: dict, seconds: float, hz: float, viser_urdf=None):
    """Smoothly ramp ALL connected arms to the sim rest pose together (cosine
    ease-in-out), so every run/restart begins from the same pose as the simulator.
    robots: {'left': robot, 'right': robot} (single-arm = one entry). Mirrors the
    motion in viser if given."""
    homes = {side: home_pose(side) for side in robots}
    starts = {}
    for side, rob in robots.items():
        try:
            starts[side] = read_joints(rob)
        except Exception as e:
            print(f"[{side.capitalize()}] Could not read joints for homing ({e}); skipping.")
    if not starts:
        return
    n = max(1, int(seconds * hz))
    print(f"[Home] ramping {' + '.join(s.capitalize() for s in starts)} to home pose "
          f"over {seconds:.0f}s — keep the workspace clear …")
    for i in range(1, n + 1):
        s = 0.5 - 0.5 * math.cos(math.pi * i / n)   # ease in-out
        view = {}
        for side, start in starts.items():
            home = homes[side]
            interp = {j: (1.0 - s) * start[j] + s * home[j] for j in JOINT_NAMES}
            try:
                robots[side].send_action({f"{j}.pos": interp[j] for j in JOINT_NAMES})
            except Exception as e:
                print(f"[{side.capitalize()}] Homing send error: {e}")
            view[side] = interp
        if viser_urdf is not None:
            try:
                viser_urdf.update_cfg(_build_view_cfg(view))
            except Exception:
                pass
        await asyncio.sleep(1.0 / hz)
    print("[Home] at rest pose.")


def _step_arm(arm_label: str, goal, mapper: VRToSO101Mapper,
              robot, prev_grip: bool, current_joints: dict, yaw_deg=None) -> bool:
    """Run one GRIP-CLUTCHED control step for one arm. Returns the new grip state.
    current_joints = the arm's actual joint angles, read once per loop by the caller.

    The arm only moves while you SQUEEZE GRIP. On the grip press it re-anchors to
    the arm's current physical pose (reads the motors), so the arm holds still and
    then moves relative to your hand from there; releasing grip freezes the arm so
    you can reposition your hand and squeeze grip again to continue (clutch, like
    lifting a mouse). Crucially this means the robot never moves unless you are
    actively gripping — no motion from incidental hand movement before you engage.

    Requires gripActive forwarded in the goal metadata (additive edit in
    XLeVR/xlevr/inputs/vr_ws_server.py). If a build of XLeVR doesn't forward it,
    grip is always False and the arm simply won't move — fail-safe, not run-away.
    """
    if goal is None:
        return prev_grip

    grip = bool((goal.metadata or {}).get('grip_active', False))

    if grip and not prev_grip:
        mapper.reset(robot)            # anchor at the arm's current physical pose
    elif prev_grip and not grip:
        print(f"[{arm_label}] grip released — arm holding position.")

    if grip and goal.target_position is not None:
        mapper.handle_vr_input(goal, headset_yaw_deg=yaw_deg)
        try:
            action = mapper.get_action(current_joints)
            robot.send_action(action)
        except Exception as e:
            print(f"[{arm_label}] Control error: {e}")

    return grip


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

    # Connect robot(s) → uniform {side: robot} map. Each arm is tried independently;
    # a failed arm is skipped with a warning so the other arm can still run.
    # robots is defined before the try/finally so cleanup always has it.
    robots: dict = {}
    _to_connect = (
        [("left",  cfg.left_robot_port,  cfg.robot_camera_index, "robot2"),
         ("right", cfg.right_robot_port, cfg.robot_camera_index, "robot1")]
        if dual else
        [(cfg.arm, cfg.robot_port, cfg.robot_camera_index, "robot1")]
    )
    for side, port, cam, rid in _to_connect:
        try:
            robots[side] = _connect_robot(port, cam, side.capitalize(), robot_id=rid)
        except Exception as e:
            print(f"[{side.capitalize()}] ⚠️  Could not connect on {port}: {e}")
            print(f"[{side.capitalize()}]    Skipping this arm — check power/USB and retry.")
    if not robots:
        print("[ERROR] No arms connected. Exiting.")
        monitor_task.cancel()
        await monitor.vr_server.stop()
        await runner.cleanup()
        if cf_proc:
            cf_proc.terminate()
        return

    # Optional viser mirror of the real robot's ACTUAL joints (created first so it
    # mirrors the startup homing too).
    viser_server = None
    viser_urdf = None
    if cfg.viz and not _VIZ_AVAILABLE:
        print("[viz] viser/yourdfpy not installed; continuing without the robot mirror.")
    elif cfg.viz:
        try:
            viser_server = viser.ViserServer(port=VIZ_PORT)
            viser_urdf = viser.extras.ViserUrdf(
                viser_server, yourdfpy.URDF.load(URDF_PATH), root_node_name="/robot")
            try:
                viser_urdf.update_cfg(_build_view_cfg({s: read_joints(r) for s, r in robots.items()}))
            except Exception:
                viser_urdf.update_cfg(_build_view_cfg({}))
            print(f"🤖 Robot mirror: open http://localhost:{VIZ_PORT} on this computer "
                  f"to watch the real robot.")
        except Exception as e:
            print(f"[viz] Could not start viser ({e}); continuing without mirror.")
            viser_server = viser_urdf = None

    # Home to the sim's fixed rest pose so every run starts from the same place.
    if cfg.home_on_start:
        await _home_arms(robots, cfg.home_seconds, cfg.control_hz, viser_urdf)

    print(f"\n[Info] mode={'dual-arm' if dual else cfg.arm}  kp={cfg.kp}  hz={cfg.control_hz}")
    print("[Info] SAFETY: each arm only moves while you SQUEEZE GRIP on that controller.")
    print("[Info]   grip ON  → anchors at the arm's current pose, then your hand moves it.")
    print("[Info]   grip OFF → arm freezes; reposition your hand and grip again. (clutch)")
    print("[Info]   DOUBLE-CLICK grip → return to home pose and start over.")
    print("[Info]   trigger  → gripper.   Ctrl+C → home, then quit.\n")

    dt = 1.0 / cfg.control_hz
    pan_sign = {"left": +1.0, "right": +1.0}
    mappers = {s: VRToSO101Mapper(cfg, pan_sign=pan_sign.get(s, +1.0)) for s in robots}
    grip    = {s: False for s in robots}

    # Double-click-grip detection (two grip presses within the window → restart).
    DOUBLE_CLICK_WINDOW = 0.4
    last_press = {s: -1e9 for s in robots}
    raw_prev   = {s: False for s in robots}

    try:
        while True:
            t_start = time.perf_counter()

            yaw = headset_yaw_deg(monitor.get_latest_goal_nowait("headset"))
            goals = {s: monitor.get_latest_goal_nowait(s) for s in robots}

            # Read every arm's ACTUAL joints once per loop (control + mirror).
            # A transient bus error just drops a frame.
            try:
                joints = {s: read_joints(r) for s, r in robots.items()}
            except Exception as e:
                print(f"[Loop] joint read error: {e}")
                await asyncio.sleep(dt)
                continue

            # Detect a grip double-click on any controller → restart (home + fresh).
            restart = False
            for s in robots:
                g = goals[s]
                raw = bool((g.metadata or {}).get('grip_active', False)) if g else False
                if raw and not raw_prev[s]:                       # grip press (rising edge)
                    if (t_start - last_press[s]) < DOUBLE_CLICK_WINDOW:
                        restart = True
                    last_press[s] = t_start
                raw_prev[s] = raw

            if restart:
                print("[Restart] grip double-click → homing and starting over.")
                await _home_arms(robots, cfg.home_seconds, cfg.control_hz, viser_urdf)
                for s in robots:                                  # fresh start, like a rerun
                    grip[s] = False
                    last_press[s] = -1e9
                    g = goals[s]
                    raw_prev[s] = bool((g.metadata or {}).get('grip_active', False)) if g else False
                continue

            for s in robots:
                grip[s] = _step_arm(s.capitalize(), goals[s], mappers[s],
                                    robots[s], grip[s], joints[s], yaw)
            if viser_urdf is not None:
                viser_urdf.update_cfg(_build_view_cfg(joints))

            elapsed = time.perf_counter() - t_start
            await asyncio.sleep(max(0.0, dt - elapsed))

    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n[Info] Shutting down — returning arms to home first …")
    finally:
        monitor_task.cancel()
        await monitor.vr_server.stop()
        await runner.cleanup()
        # Return the arms to the home pose before disconnecting.
        try:
            await _home_arms(robots, cfg.home_seconds, cfg.control_hz, viser_urdf)
        except Exception as e:
            print(f"[Home] shutdown homing skipped: {e}")
        if viser_server is not None:
            try:
                viser_server.stop()
            except Exception:
                pass
        for side, robot in robots.items():
            try:
                robot.disconnect()
            except Exception as e:
                print(f"[{side.capitalize()}] Disconnect error (motor may be in overload): {e}")
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
    p.add_argument("--robot_port",       default="/dev/tty.usbmodemXXXX",
                   help="Port for single-arm mode (--arm left or right)")
    p.add_argument("--left_robot_port",  default="/dev/tty.usbmodemXXXX",
                   help="Left arm port (used when --arm both)")
    p.add_argument("--right_robot_port", default="/dev/tty.usbmodemXXXX",
                   help="Right arm port (used when --arm both)")
    p.add_argument("--control_hz",       type=float, default=30.0)
    p.add_argument("--kp",               type=float, default=1.0,
                   help="P-control gain (1.0=direct, lower=smoother but laggier)")
    p.add_argument("--no-home",          dest="home_on_start", action="store_false",
                   help="Skip the startup move to the sim rest pose (arm stays where it is).")
    p.add_argument("--home_seconds",     type=float, default=2.0,
                   help="Duration of the startup homing ramp (default 2s).")
    p.add_argument("--no-viz",           dest="viz", action="store_false",
                   help="Skip the viser mirror of the real robot (http://localhost:8081).")
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
        home_on_start=args.home_on_start,
        home_seconds=args.home_seconds,
        viz=args.viz,
    )
    asyncio.run(run(cfg))
