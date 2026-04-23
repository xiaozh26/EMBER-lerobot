# EMBER-Lerobot — VR Teleoperation for SO-101

> A project by the **Embodied Intelligence & Robotics Center (EMBER), UC Berkeley**

> Control a real SO-101 robot arm with a **Meta Quest 3** headset in real time.
> Delta-control IK bridge built on top of [LeRobot](https://github.com/huggingface/lerobot) and [XLeRobot/XLeVR](https://github.com/Vector-Wangel/XLeRobot).

---

## Overview

This project adds a VR teleoperation layer on top of the standard LeRobot stack. Instead of replaying recorded trajectories, you stream live hand-tracking data from a Quest 3 to a SO-101 arm (or a pair of arms) over a Cloudflare tunnel — no local network pairing required.

**Key design choices:**

| Feature | Detail |
|---|---|
| Control mode | Frame-to-frame delta (no jump on connect) |
| Arm kinematics | 2D IK in the vertical plane with α=0.1 smoothing |
| Wrist compensation | Auto-levels end-effector orientation as shoulder/elbow move |
| Gripper | Trigger button → binary open/close |
| Latency path | Quest 3 browser → Cloudflare tunnel → local aiohttp server → robot |
| Supported configs | Single arm (left or right) · Dual arm simultaneous |

---

## Prerequisites

### 1. Install LeRobot

Follow the [official LeRobot installation guide](https://github.com/huggingface/lerobot?tab=readme-ov-file#installation) and make sure the package is importable:

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e ".[feetech]"
```

Verify:

```bash
python -c "import lerobot; print('LeRobot OK')"
```

### 2. Clone XLeRobot (for XLeVR web UI and VRMonitor)

The VR bridge depends on `VRMonitor` and the Quest 3 web UI from XLeRobot:

```bash
# Clone into the same parent directory as this repo
git clone https://github.com/Vector-Wangel/XLeRobot.git
```

Expected layout:

```
parent_dir/
├── EMBER-lerobot/     ← this repo
└── XLeRobot/
    └── XLeVR/
        ├── vr_monitor.py
        └── web-ui/
```

### 3. Install Cloudflared

Cloudflared creates a public HTTPS tunnel so the Quest 3 browser can reach your local server without any network configuration.

```bash
brew install cloudflared      # macOS
# or: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/
```

### 4. Install Python dependencies

```bash
pip install aiohttp
```

---

## Hardware Setup

1. Connect your SO-101 arm(s) via USB.
2. Find the serial port(s):
   ```bash
   ls /dev/tty.usbmodem*
   ```
3. Run the LeRobot motor calibration if you haven't already:
   ```bash
   python -m lerobot.scripts.configure_motor --port /dev/tty.usbmodemXXXX
   ```

---

## Usage

### Single arm

```bash
python src/lerobot/scripts/vr_teleop_real.py \
    --arm right \
    --robot_port /dev/tty.usbmodem5A7C1231311
```

### Dual arm

```bash
python src/lerobot/scripts/vr_teleop_real.py \
    --arm both \
    --left_robot_port  /dev/tty.usbmodem5A7C1190111 \
    --right_robot_port /dev/tty.usbmodem5A7C1231311
```

### All CLI flags

| Flag | Default | Description |
|---|---|---|
| `--arm` | `both` | `left` · `right` · `both` |
| `--robot_port` | — | Serial port for single-arm mode |
| `--left_robot_port` | — | Serial port for left arm (dual mode) |
| `--right_robot_port` | — | Serial port for right arm (dual mode) |
| `--control_hz` | `30.0` | Control loop frequency (Hz) |
| `--kp` | `1.0` | P-gain: 1.0 = direct, lower = smoother but laggier |

---

## Connecting the Quest 3

Once the script is running you will see output like:

```
🌐 Combined server on port 8080
┌──────────────────────────────────────┐
│  Quest 3 browser → type this:        │
│  http://192.168.x.x:8080/go          │
│  It will redirect to the VR page.    │
│  Then tap the VR goggles icon.       │
└──────────────────────────────────────┘
```

1. On the Quest 3, open the **Meta browser**.
2. Type the local URL shown in the terminal (e.g. `http://192.168.x.x:8080/go`). It redirects through the Cloudflare tunnel to the VR web UI.
3. Tap the **VR goggles icon** to enter immersive mode.
4. The arm will **not move** until the Quest 3 session is active — on first position data the arm seeds its target from the current joint angles and holds in place.

---

## How It Works

```
Quest 3 (WebXR)
    │  hand pose @ ~72 Hz
    ▼
Cloudflare Tunnel  ──►  aiohttp server (port 8080)
                              │
                         WS proxy
                              │
                         VRMonitor (XLeVR)
                              │
                         VRToSO101Mapper
                           ├─ frame delta → shoulder_pan
                           ├─ 2D IK (reach, height) → shoulder_lift + elbow_flex
                           ├─ wrist level compensation → wrist_flex
                           ├─ VR rotation delta → wrist_roll
                           └─ trigger → gripper
                              │
                         P-control @ 30 Hz
                              │
                         SO-101 motors (Feetech serial)
```

**Delta control** means only the *change* in VR position between frames drives the arm, so connecting mid-motion never causes a jump.

**Wrist level compensation** keeps the end-effector pointing forward regardless of arm pose by solving:

```
wrist_flex = pitch_accumulator − shoulder_lift − elbow_flex
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Could not import VRMonitor` | Check that `XLeRobot/XLeVR/` exists at the expected path relative to this repo |
| `cloudflared` not found | `brew install cloudflared` |
| Arm jumps on first VR frame | Already handled — first frame only sets the baseline, no motion |
| Arm drifts slowly | Lower `--kp` (e.g. `0.5`) or check USB cable quality |
| Quest browser shows blank page | Wait a few seconds for the Cloudflare tunnel to initialize |
| Motor overload error on disconnect | Normal — the script catches and reports it cleanly |

---

## Acknowledgements

Built on top of:
- [LeRobot](https://github.com/huggingface/lerobot) by Hugging Face
- [XLeRobot / XLeVR](https://github.com/Vector-Wangel/XLeRobot) by Vector-Wangel

---

## License

This project's additions are released under the Apache 2.0 License.
The underlying LeRobot code retains its original Apache 2.0 License.
