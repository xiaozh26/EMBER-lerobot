"""
Simple URDF viewer with joint sliders.

Usage:
    python view_robot.py
    # Then open http://localhost:8082 in your browser.
"""

import os
import time

import numpy as np
import viser
import viser.extras
import yourdfpy

URDF_PATH = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../../../XLeRobot/simulation/Maniskill/assets/xlerobot/xlerobot.urdf",
))

# Actuated DOF order as returned by yourdfpy (matches update_cfg array index):
#  0: root_x_axis_joint   1: root_y_axis_joint   2: root_z_rotation_joint
#  3: Rotation  4: Pitch  5: Elbow  6: Wrist_Pitch  7: Wrist_Roll  8: Jaw
#  9: Rotation_2  10: Pitch_2  11: Elbow_2  12: Wrist_Pitch_2
# 13: Wrist_Roll_2  14: Jaw_2   15: head_pan_joint  16: head_tilt_joint

# (label, dof_index, lower_rad, upper_rad)
RIGHT_ARM = [
    ("Shoulder Pan",  3,  -2.10,  2.10),
    ("Shoulder Lift", 4,  -0.10,  3.45),
    ("Elbow",         5,  -0.20,  3.14),
    ("Wrist Pitch",   6,  -1.80,  1.80),
    ("Wrist Roll",    7,  -3.14,  3.14),
    ("Gripper",       8,   0.00,  1.70),
]
LEFT_ARM = [
    ("Shoulder Pan",  9,  -2.10,  2.10),
    ("Shoulder Lift", 10, -0.10,  3.45),
    ("Elbow",         11, -0.20,  3.14),
    ("Wrist Pitch",   12, -1.80,  1.80),
    ("Wrist Roll",    13, -3.14,  3.14),
    ("Gripper",       14,  0.00,  1.70),
]
HEAD = [
    ("Pan",  15, -1.57, 1.57),
    ("Tilt", 16, -0.76, 1.45),
]


def main():
    server = viser.ViserServer(port=8082)
    print("[Viser] Open http://localhost:8082 in your browser.")

    urdf = yourdfpy.URDF.load(URDF_PATH)
    viser_urdf = viser.extras.ViserUrdf(server, urdf, root_node_name="/robot")

    sliders: dict[int, viser.GuiSliderHandle] = {}

    with server.gui.add_folder("Right Arm"):
        for label, idx, lo, hi in RIGHT_ARM:
            sliders[idx] = server.gui.add_slider(label, lo, hi, step=0.01, initial_value=0.0)

    with server.gui.add_folder("Left Arm"):
        for label, idx, lo, hi in LEFT_ARM:
            sliders[idx] = server.gui.add_slider(label, lo, hi, step=0.01, initial_value=0.0)

    with server.gui.add_folder("Head"):
        for label, idx, lo, hi in HEAD:
            sliders[idx] = server.gui.add_slider(label, lo, hi, step=0.01, initial_value=0.0)

    cfg = np.zeros(17)
    while True:
        for idx, slider in sliders.items():
            cfg[idx] = slider.value
        viser_urdf.update_cfg(cfg)
        time.sleep(1 / 30)


if __name__ == "__main__":
    main()
