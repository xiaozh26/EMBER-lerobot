"""
VR Teleoperation → XLeRobot Viser Simulation
============================================
Drive the XLeRobot URDF in a viser scene with a Meta Quest 3. Move the
controller and the simulated arm follows the controller's position + rotation.

Same control brain as vr_teleop_real.py (origin-relative 2D IK + wrist
level-compensation), but instead of a physical SO-101 it updates a viser URDF.
The SO-101 *servo* degrees the mapper produces are converted to the *raw URDF
radians* the xlerobot model expects (verified bit-exact vs the ManiSkill VR
demo's inverse_kinematics):
    shoulder_lift_servo →  Pitch_urdf  = rad(90 - s)
    elbow_flex_servo    →  Elbow_urdf  = rad(e + 90)
    wrist_flex_servo    →  Wrist_Pitch = rad(w)        (level-comp algebra cancels)
    shoulder_pan/roll   →  direct rad  (× sign knobs below if mirrored)
    gripper             →  Jaw_urdf    = rad(g)
Servo-neutral (all 0) → Pitch = Elbow = π/2 → upright rest pose.

This script is self-contained: it needs only viser, yourdfpy, aiohttp, numpy
and XLeRobot/XLeVR (vr_monitor) — NOT the real-robot driver stack.

Usage:
    python vr_teleop_sim.py                 # both arms
    python vr_teleop_sim.py --arm right     # single arm

  1. Watch the sim on your computer:  http://localhost:8081
  2. A cloudflared tunnel for the Quest is started automatically. In the Quest 3
     browser, type the short  http://<your-ip>:8080/go  URL it prints — it
     redirects to the VR page. Tap the VR goggles icon. (If cloudflared isn't
     installed:  brew install cloudflared  then run manually:
        cloudflared tunnel --url http://localhost:8080 )

Control = GRIP CLUTCH. The arm stays at the loaded rest pose until you squeeze
GRIP. Grip press anchors the origin at the current pose WITHOUT moving (so the
first grip starts exactly from the loaded rest pose); while held, the hand moves
the arm relative to that anchor; releasing grip freezes the arm so you can
reposition your hand and grip again (like lifting a mouse). The trigger controls
the gripper. This needs gripActive forwarded in the goal metadata — see the
one-line additive edit in XLeVR/xlevr/inputs/vr_ws_server.py.

Motion mapping (fixed WORLD frame by default):
    push forward/back → reach   (shoulder + ELBOW bend)
    move right/left   → base pan
    move up/down      → height
    rotate controller → wrist pitch / roll
World frame means "forward" is fixed to the play space, so a forward push always
drives the elbow. (The old body-relative remap rotated motion by headset yaw and,
when you faced off-axis to watch the desktop sim, sent the whole forward push into
pan — leaving the elbow dead. Pass --body-relative to opt back in.)
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
import numpy as np
import viser
import viser.extras
import yourdfpy
from aiohttp import web

# ── Paths ────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
XLEVR_PATH   = os.path.normpath(os.path.join(_HERE, "../../../../XLeRobot/XLeVR"))
XLEVR_WEB_UI = os.path.join(XLEVR_PATH, "web-ui")
URDF_PATH = os.path.normpath(os.path.join(
    _HERE, "../../../../XLeRobot/simulation/Maniskill/assets/xlerobot/xlerobot.urdf"))

COMBINED_PORT = 8080   # VR web page + WS proxy (this is the cloudflared target)
VISER_PORT    = 8081   # sim view you open on your own computer

if XLEVR_PATH not in sys.path:
    sys.path.insert(0, XLEVR_PATH)

try:
    from vr_monitor import VRMonitor, get_local_ip
except ImportError as e:
    print(f"[ERROR] Could not import VRMonitor from {XLEVR_PATH}: {e}")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Servo → URDF conversion knobs (flip if an axis looks mirrored in the sim)
# ─────────────────────────────────────────────────────────────────────────────
_PAN_SIGN_URDF  = 1.0    # shoulder_pan → URDF Rotation sign
_ROLL_SIGN_URDF = 1.0    # wrist_roll   → URDF Wrist_Roll sign

# Actuated DOF order returned by yourdfpy (index into the update_cfg array):
#  0 root_x  1 root_y  2 root_z   3 Rotation  4 Pitch  5 Elbow  6 Wrist_Pitch
#  7 Wrist_Roll  8 Jaw   9 Rotation_2 10 Pitch_2 11 Elbow_2 12 Wrist_Pitch_2
# 13 Wrist_Roll_2 14 Jaw_2   15 head_pan  16 head_tilt
ARM_DOF = {
    # joint name        → URDF dof index for (right, left)
    "shoulder_pan":  (3, 9),
    "shoulder_lift": (4, 10),
    "elbow_flex":    (5, 11),
    "wrist_flex":    (6, 12),
    "wrist_roll":    (7, 13),
    "gripper":       (8, 14),
}
JOINT_NAMES = list(ARM_DOF.keys())
N_DOF = 17

# Home / parking pose (servo degrees), per arm — measured on the real arms; kept in
# sync with vr_teleop_real.py HOME_POSE. This is a folded/parked configuration
# (shoulder_lift≈-99, elbow_flex≈97). That elbow is beyond the reach-IK's range, so
# the sim loads this pose, and squeezing grip UNFOLDS the arm into the teleop
# workspace and then tracks your hand.
HOME_POSE = {
    "left":  {"shoulder_pan": -7.60, "shoulder_lift": -99.47, "elbow_flex": 97.98,
              "wrist_flex": -2.11, "wrist_roll": -14.64, "gripper": 0.0},
    "right": {"shoulder_pan": -3.30, "shoulder_lift": -99.03, "elbow_flex": 96.35,
              "wrist_flex":  2.73, "wrist_roll": -178.07, "gripper": 0.0},
}


def servo_deg_to_urdf_rad(servo: dict) -> dict:
    """SO-101 servo degrees → raw xlerobot URDF radians (see module docstring)."""
    return {
        "shoulder_pan":  _PAN_SIGN_URDF  * math.radians(servo["shoulder_pan"]),
        "shoulder_lift": math.radians(90.0 - servo["shoulder_lift"]),
        "elbow_flex":    math.radians(servo["elbow_flex"] + 90.0),
        "wrist_flex":    math.radians(servo["wrist_flex"]),
        "wrist_roll":    _ROLL_SIGN_URDF * math.radians(servo["wrist_roll"]),
        "gripper":       math.radians(servo["gripper"]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Minimal SO-101 2-link planar kinematics (copied from lerobot SO101Robot so the
# sim doesn't need the heavy robot driver stack). Matches it bit-for-bit.
# ─────────────────────────────────────────────────────────────────────────────

class SO101Kinematics:
    def __init__(self, l1=0.1159, l2=0.1350):
        self.l1 = l1
        self.l2 = l2

    def inverse_kinematics(self, x, y, j2_lower=-0.1):
        # j2_lower = lower clamp on the shoulder Pitch (rad). The ManiSkill URDF
        # declares -0.1, but the real SO-arm pitches further down (it can pick
        # objects below the base), so the sim relaxes this to reach down in front.
        l1, l2 = self.l1, self.l2
        theta1_offset = math.atan2(0.028, 0.11257)
        theta2_offset = math.atan2(0.0052, 0.1349) + theta1_offset

        r = math.sqrt(x**2 + y**2)
        r_max = l1 + l2
        if r > r_max:
            s = r_max / r; x *= s; y *= s; r = r_max
        r_min = abs(l1 - l2)
        if r < r_min and r > 0:
            s = r_min / r; x *= s; y *= s; r = r_min

        cos_theta2 = -(r**2 - l1**2 - l2**2) / (2 * l1 * l2)
        cos_theta2 = max(-1.0, min(1.0, cos_theta2))
        theta2 = math.pi - math.acos(cos_theta2)
        beta = math.atan2(y, x)
        gamma = math.atan2(l2 * math.sin(theta2), l1 + l2 * math.cos(theta2))
        theta1 = beta + gamma

        joint2 = max(j2_lower, min(3.45, theta1 + theta1_offset))
        joint3 = max(-0.2, min(math.pi, theta2 + theta2_offset))
        # servo coordinate transform
        return 90 - math.degrees(joint2), math.degrees(joint3) - 90

    def forward_kinematics(self, joint2_deg, joint3_deg):
        l1, l2 = self.l1, self.l2
        joint2_rad = math.radians(90 - joint2_deg)
        joint3_rad = math.radians(joint3_deg + 90)
        theta1_offset = math.atan2(0.028, 0.11257)
        theta2_offset = math.atan2(0.0052, 0.1349) + theta1_offset
        theta1 = joint2_rad - theta1_offset
        theta2 = joint3_rad - theta2_offset
        x = l1 * math.cos(theta1) + l2 * math.cos(theta1 + theta2 - math.pi)
        y = l1 * math.sin(theta1) + l2 * math.sin(theta1 + theta2 - math.pi)
        return x, y


def headset_yaw_deg(headset_goal) -> Optional[float]:
    """Headset yaw (deg). XLeVR packs euler-Y into the goal's wrist_roll_deg."""
    if headset_goal is None:
        return None
    return getattr(headset_goal, "wrist_roll_deg", None)


@dataclass
class VRTeleopConfig:
    arm: str = "both"        # "left" | "right" | "both"
    control_hz: float = 30.0
    gripper_trigger_threshold: float = 0.5
    use_headset_yaw: bool = False   # False = fixed world frame (forward push → reach);
                                    # True  = body-relative (rotate motion by headset yaw)
    level_wrist: bool = False       # False = gripper follows the forearm (reaches down to pick);
                                    # True  = auto-keep the gripper world-level


# ─────────────────────────────────────────────────────────────────────────────
# Mapper — origin-relative control (identical to vr_teleop_real.py VRToSO101Mapper)
# ─────────────────────────────────────────────────────────────────────────────

class VRToSO101Mapper:
    """Origin-relative control: the hand pose at grip is the origin; every frame
    the arm target is recomputed from the hand's displacement-from-origin (not
    accumulated), so a given hand pose always maps to the same arm pose and the
    arm returns exactly even after hitting a workspace limit. Outputs are
    alpha-smoothed to reject VR tracking glitches."""

    _REACH_GAIN   = 1.6     # m reach  per m of hand-forward (was 1.2; reaches max faster)
    _HEIGHT_GAIN  = 2.2     # m height per m of hand-up      (was 1.2; reach the low workspace
                            #                                 without dropping your hand to the floor)
    _PAN_GAIN     = 510.0
    _ANGLE_SCALE  = 3.0
    _ROLL_DELTA_LIMIT = 90.0
    _ALPHA        = 0.1
    _HEADSET_YAW_SIGN = 1.0
    # Shoulder Pitch lower clamp (rad). ManiSkill URDF says -0.1, but the real arm
    # pitches well past it to pick objects below the base — relax so the sim can
    # reach down in front of the cart. Less-negative = stays higher/more physical.
    _PITCH_MIN_RAD = -0.9
    _Y_MIN        = -0.35   # lowest commanded reach height (was -0.15, which the old
                            #                                Pitch cap made unreachable anyway)
    _Y_MAX        =  0.22

    def __init__(self, cfg: VRTeleopConfig, pan_sign: float = 1.0, home: dict = None):
        self.cfg = cfg
        self.ik = SO101Kinematics()
        self.pan_sign = pan_sign
        self.home = dict(home) if home else None

        # Reach state where teleop anchors after the arm "wakes up" from the parked
        # home (IK-consistent so it doesn't drift once moving).
        self.current_x = 0.1629
        self.current_y = 0.1131
        self.pitch = 0.0

        self.target_positions = {j: 0.0 for j in JOINT_NAMES}

        self.origin_vr_pos = None
        self.origin_yaw = None
        self.origin_wrist_flex_deg = None
        self.origin_wrist_roll_deg = None
        self.base_x = self.current_x
        self.base_y = self.current_y
        self.base_pan = 0.0
        self.base_pitch = 0.0
        self.base_roll = 0.0
        self.base_wrist_flex = 0.0
        self._dbg_count = 0

        # Ready reach (IK-consistent) — pitch baseline for the level-comp wrist mode.
        lift0, elbow0 = self.ik.inverse_kinematics(self.current_x, self.current_y,
                                                   j2_lower=self._PITCH_MIN_RAD)
        self.pitch = lift0 + elbow0
        if self.home is not None:
            # Load the folded/parked home pose for display. Its elbow is past the
            # reach-IK range, so the first grip unfolds the arm toward the ready
            # reach (current_x/y) — the IK + alpha smoothing ramps it in smoothly.
            for j in JOINT_NAMES:
                self.target_positions[j] = float(self.home.get(j, 0.0))
        else:
            # No home given → seed the IK-consistent rest directly (legacy behavior).
            self.target_positions["shoulder_lift"] = lift0
            self.target_positions["elbow_flex"]    = elbow0
            self.target_positions["wrist_flex"]    = 0.0

    def reanchor(self):
        """Grip press: re-capture the VR origin on the next gripped frame WITHOUT
        moving the arm. handle_vr_input grabs the base pose from the current
        target state on that frame, so the arm holds exactly where it is (the
        loaded rest pose on the very first grip) and moves relative from there."""
        self.origin_vr_pos = None
        self.origin_yaw = None
        self.origin_wrist_flex_deg = None
        self.origin_wrist_roll_deg = None

    def handle_vr_input(self, goal, headset_yaw_deg=None):
        if goal is None or goal.target_position is None:
            return

        current_vr_pos = goal.target_position

        if self.origin_vr_pos is None:
            self.origin_vr_pos = list(current_vr_pos)
            self.origin_yaw = headset_yaw_deg
            self.base_x = self.current_x
            self.base_y = self.current_y
            self.base_pan = self.target_positions["shoulder_pan"]
            self.base_pitch = self.pitch
            self.base_roll = self.target_positions["wrist_roll"]
            self.base_wrist_flex = self.target_positions["wrist_flex"]
            self.origin_wrist_flex_deg = getattr(goal, "wrist_flex_deg", None)
            self.origin_wrist_roll_deg = getattr(goal, "wrist_roll_deg", None)
            return

        disp_x = current_vr_pos[0] - self.origin_vr_pos[0]
        disp_y = current_vr_pos[1] - self.origin_vr_pos[1]
        disp_z = current_vr_pos[2] - self.origin_vr_pos[2]

        # Map hand displacement → forward(reach) / right(pan).
        # WORLD frame (default): forward = −Z, right = +X, fixed to the play space.
        # This guarantees a forward push drives reach → shoulder/elbow. The optional
        # body-relative remap (rotate by headset yaw) sends a forward push into pan
        # when you face off-axis — which in the sim (you watch the desktop, not the
        # headset) left the elbow "dead". Enable with --body-relative only if you
        # operate while facing the same way you did when the page loaded.
        if self.cfg.use_headset_yaw and self.origin_yaw is not None:
            th = math.radians(self.origin_yaw) * self._HEADSET_YAW_SIGN
            s, c = math.sin(th), math.cos(th)
            forward = -(disp_x * s + disp_z * c)
            right = (disp_x * c - disp_z * s)
        else:
            forward = -disp_z
            right = disp_x

        MAX_REACH = self.ik.l1 + self.ik.l2 - 0.003
        self.current_x = max(0.04, min(MAX_REACH, self.base_x + forward * self._REACH_GAIN))
        self.current_y = max(self._Y_MIN, min(self._Y_MAX, self.base_y + disp_y * self._HEIGHT_GAIN))

        a = self._ALPHA

        pan_target = max(-90.0, min(90.0,
            self.base_pan + right * self._PAN_GAIN * self.pan_sign))
        self.target_positions["shoulder_pan"] = (
            (1 - a) * self.target_positions["shoulder_pan"] + a * pan_target)

        try:
            joint2, joint3 = self.ik.inverse_kinematics(self.current_x, self.current_y,
                                                        j2_lower=self._PITCH_MIN_RAD)
            self.target_positions["shoulder_lift"] = (
                (1 - a) * self.target_positions["shoulder_lift"] + a * joint2)
            self.target_positions["elbow_flex"] = (
                (1 - a) * self.target_positions["elbow_flex"] + a * joint3)
        except Exception as e:
            print(f"[Mapper] IK failed (x={self.current_x:.3f}, y={self.current_y:.3f}): {e}")

        # User's controller pitch relative to grip (deg, 0 at grip).
        user_pitch = 0.0
        if getattr(goal, "wrist_flex_deg", None) is not None:
            if self.origin_wrist_flex_deg is None:
                self.origin_wrist_flex_deg = goal.wrist_flex_deg
            user_pitch = (goal.wrist_flex_deg - self.origin_wrist_flex_deg) * self._ANGLE_SCALE

        if self.cfg.level_wrist:
            # Keep the gripper world-level regardless of arm pose, + user pitch.
            # (Good for carrying things level, but it curls the gripper UP when you
            #  reach down, so the tip can't descend — bad for picking from below.)
            self.pitch = max(-90.0, min(90.0, self.base_pitch + user_pitch))
            self.target_positions["wrist_flex"] = (
                -self.target_positions["shoulder_lift"]
                - self.target_positions["elbow_flex"]
                + self.pitch)
        else:
            # Gripper FOLLOWS the forearm (default): reach down → gripper points
            # down → the tip descends onto the object. User pitch tilts it from
            # in-line. Held relative to the per-grip baseline so re-grips don't jump.
            self.target_positions["wrist_flex"] = max(-110.0, min(110.0,
                self.base_wrist_flex + user_pitch))

        if getattr(goal, "wrist_roll_deg", None) is not None:
            if self.origin_wrist_roll_deg is None:
                self.origin_wrist_roll_deg = goal.wrist_roll_deg
            roll_delta = (goal.wrist_roll_deg - self.origin_wrist_roll_deg) * self._ANGLE_SCALE
            roll_delta = max(-self._ROLL_DELTA_LIMIT, min(self._ROLL_DELTA_LIMIT, roll_delta))
            self.target_positions["wrist_roll"] = self.base_roll + roll_delta

        trigger = (goal.metadata or {}).get('trigger', 0)
        self.target_positions["gripper"] = 45.0 if trigger > self.cfg.gripper_trigger_threshold else 0.0

        self._dbg_count += 1
        if self._dbg_count % 30 == 0:
            frame = (f"body{self.origin_yaw:+.0f}°"
                     if self.cfg.use_headset_yaw and self.origin_yaw is not None else "world")
            print(f"[DBG] fwd={forward:+.3f} right={right:+.3f} up={disp_y:+.3f} "
                  f"frame={frame} → pan={self.target_positions['shoulder_pan']:+.1f}° "
                  f"lift={self.target_positions['shoulder_lift']:+.1f}° "
                  f"elbow={self.target_positions['elbow_flex']:+.1f}° "
                  f"wrist_roll={self.target_positions['wrist_roll']:+.1f}°")


# ─────────────────────────────────────────────────────────────────────────────
# Combined HTTP + WebSocket proxy (serves the VR page, proxies WS to VRMonitor)
# ─────────────────────────────────────────────────────────────────────────────

_cloudflare_url: str = ""

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
                        elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                            break

                async def u2c():
                    async for msg in up:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            await client_ws.send_str(msg.data)
                        elif msg.type == aiohttp.WSMsgType.BINARY:
                            await client_ws.send_bytes(msg.data)
                        elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
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
# Cloudflared auto-start + short URL for the Quest browser
# ─────────────────────────────────────────────────────────────────────────────

async def start_cloudflared(port: int) -> Optional[subprocess.Popen]:
    cf_bin = "cloudflared"
    try:
        subprocess.run([cf_bin, "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("[cloudflared] Not found — install with: brew install cloudflared")
        print(f"[cloudflared] Then run manually: cloudflared tunnel --url http://localhost:{port}")
        return None

    proc = subprocess.Popen(
        [cf_bin, "tunnel", "--url", f"http://localhost:{port}"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

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
        local_ip = get_local_ip()
        local_url = f"http://{local_ip}:{port}/go"
        print()
        print("┌──────────────────────────────────────┐")
        print("│  Quest 3 browser → type this:        │")
        print(f"│  {local_url:<36s}  │")
        print("│  It redirects to the VR page; then   │")
        print("│  tap the VR goggles icon & squeeze   │")
        print("│  grip to start controlling.          │")
        print("└──────────────────────────────────────┘")
        print()
    else:
        print("[cloudflared] Could not detect URL — check terminal output")

    return proc


# ─────────────────────────────────────────────────────────────────────────────
# Per-arm control step
# ─────────────────────────────────────────────────────────────────────────────

def _step_arm(label, goal, mapper, prev_grip, yaw_deg=None) -> bool:
    """Grip-clutched control step. Returns the new grip state.

    The arm holds its current pose until you squeeze GRIP. On the grip press it
    anchors the origin to the current hand pose WITHOUT moving (so the first grip
    starts exactly from the loaded rest pose); while grip is held the arm moves
    relative to that anchor; releasing grip freezes the arm so you can reposition
    your hand and squeeze again to continue (clutch, like lifting a mouse)."""
    if goal is None:
        return prev_grip

    grip = bool((goal.metadata or {}).get("grip_active", False))

    if grip and not prev_grip:
        mapper.reanchor()
        print(f"[{label}] grip ON — anchored; move to teleop (release to reposition).")
    elif prev_grip and not grip:
        print(f"[{label}] grip OFF — arm frozen.")

    if grip and goal.target_position is not None:
        mapper.handle_vr_input(goal, headset_yaw_deg=yaw_deg)

    return grip


def build_cfg(mappers: dict) -> np.ndarray:
    """Render BOTH arms from their mapper target state → 17-DOF URDF radians."""
    cfg = np.zeros(N_DOF)
    for side, mapper in mappers.items():
        side_idx = 1 if side == "left" else 0
        urdf = servo_deg_to_urdf_rad(mapper.target_positions)
        for joint, (r_idx, l_idx) in ARM_DOF.items():
            cfg[l_idx if side_idx else r_idx] = urdf[joint]
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

async def run(cfg: VRTeleopConfig):
    arms = ["left", "right"] if cfg.arm == "both" else [cfg.arm]

    # ── Viser sim ────────────────────────────────────────────────────────────
    server = viser.ViserServer(port=VISER_PORT)
    urdf = yourdfpy.URDF.load(URDF_PATH)
    viser_urdf = viser.extras.ViserUrdf(server, urdf, root_node_name="/robot")
    print(f"\n🤖 Sim view: open http://localhost:{VISER_PORT} on this computer.")

    # Always render both arms (each mapper starts at the loaded rest pose); only
    # the selected arm(s) receive VR input — the rest stay frozen at rest.
    mappers = {"left":  VRToSO101Mapper(cfg, pan_sign=1.0, home=HOME_POSE["left"]),
               "right": VRToSO101Mapper(cfg, pan_sign=1.0, home=HOME_POSE["right"])}
    viser_urdf.update_cfg(build_cfg(mappers))  # show the loaded rest pose immediately

    # ── VR monitor (WebSocket server) ────────────────────────────────────────
    monitor = VRMonitor()
    if not monitor.initialize():
        print("[ERROR] Failed to initialize VRMonitor")
        return
    await monitor.vr_server.start()
    monitor.is_running = True

    ws_port = monitor.config.websocket_port
    runner = web.AppRunner(make_combined_app(ws_port))
    await runner.setup()
    await web.TCPSite(runner, "0.0.0.0", COMBINED_PORT).start()
    print(f"🌐 VR server on port {COMBINED_PORT}")

    cf_proc = await start_cloudflared(COMBINED_PORT)
    monitor_task = asyncio.create_task(monitor.monitor_commands())

    prev_grip = {a: False for a in arms}

    frame = "body-relative (headset yaw)" if cfg.use_headset_yaw else "world (fixed)"
    print(f"\n[Info] mode={cfg.arm}  hz={cfg.control_hz}  frame={frame}")
    print("[Info] Arm holds the loaded rest pose until you SQUEEZE GRIP.")
    print("[Info]   • grip ON  → anchors at the current pose, then your hand drives")
    print("[Info]     the arm relative to it:")
    print("[Info]       push forward/back → reach (shoulder+ELBOW bend)")
    print("[Info]       move right/left   → base pan")
    print("[Info]       move up/down      → height")
    print("[Info]       rotate controller → wrist pitch/roll")
    print("[Info]   • grip OFF → arm freezes; reposition your hand and grip again.")
    print("[Info]   • trigger  → gripper.   Press Ctrl+C to stop.")
    if not cfg.use_headset_yaw:
        print("[Info] (World frame: face the SAME way as when you opened the page so")
        print("[Info]  'forward' lines up. Pass --body-relative to rotate with your head.)\n")
    else:
        print()

    dt = 1.0 / cfg.control_hz
    try:
        while True:
            t0 = time.perf_counter()
            yaw = headset_yaw_deg(monitor.get_latest_goal_nowait("headset"))
            for a in arms:
                goal = monitor.get_latest_goal_nowait(a)
                prev_grip[a] = _step_arm(a.capitalize(), goal, mappers[a], prev_grip[a], yaw)
            viser_urdf.update_cfg(build_cfg(mappers))
            await asyncio.sleep(max(0.0, dt - (time.perf_counter() - t0)))
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n[Info] Shutting down …")
    finally:
        monitor_task.cancel()
        await monitor.vr_server.stop()
        await runner.cleanup()
        if cf_proc:
            cf_proc.terminate()
        print("[Sim] Stopped.")


def parse_args():
    p = argparse.ArgumentParser(description="VR teleop: Meta Quest 3 → XLeRobot viser sim")
    p.add_argument("--arm", default="both", choices=["left", "right", "both"],
                   help="Which arm to control (default: both)")
    p.add_argument("--control_hz", type=float, default=30.0)
    p.add_argument("--body-relative", dest="body_relative", action="store_true",
                   help="Rotate hand motion by headset yaw (default: fixed world frame). "
                        "Only use if you face the same way as when the page loaded.")
    p.add_argument("--level-wrist", dest="level_wrist", action="store_true",
                   help="Auto-keep the gripper world-level (default: gripper follows the "
                        "forearm so it reaches down to pick).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run(VRTeleopConfig(arm=args.arm, control_hz=args.control_hz,
                                   use_headset_yaw=args.body_relative,
                                   level_wrist=args.level_wrist)))
