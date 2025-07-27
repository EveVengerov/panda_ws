import pybullet as p
import pybullet_data
import numpy as np
import time

# Connect to simulation
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# Load plane and Panda robots
p.loadURDF("plane.urdf")
robotA = p.loadURDF("franka_panda/panda.urdf", basePosition=[0.5, -0.3, 0], useFixedBase=True)
robotB = p.loadURDF("franka_panda/panda.urdf", basePosition=[0.5,  0.3, 0], useFixedBase=True)

ee_link = 11  # Panda end-effector link

# Create static debug lines once, and store their IDs
line_ids = {
    "x": None,
    "y": None,
    "z": None
}

while True:
    p.stepSimulation()

    # Get end-effector pose
    state = p.getLinkState(robotA, ee_link, computeForwardKinematics=True)
    pos = np.array(state[4])
    orn = np.array(state[5])
    rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)

    # Axis endpoints
    x_axis = pos + 0.1 * rot[:, 0]
    y_axis = pos + 0.1 * rot[:, 1]
    z_axis = pos + 0.1 * rot[:, 2]

    # First time: create the lines
    if line_ids["x"] is None:
        line_ids["x"] = p.addUserDebugLine(pos, x_axis, [1, 0, 0], 2)
        line_ids["y"] = p.addUserDebugLine(pos, y_axis, [0, 1, 0], 2)
        line_ids["z"] = p.addUserDebugLine(pos, z_axis, [0, 0, 1], 2)
    else:
        # Update line positions using replaceItemUniqueId (PyBullet >= 3.0.8)
        p.addUserDebugLine(pos, x_axis, [1, 0, 0], 2, replaceItemUniqueId=line_ids["x"])
        p.addUserDebugLine(pos, y_axis, [0, 1, 0], 2, replaceItemUniqueId=line_ids["y"])
        p.addUserDebugLine(pos, z_axis, [0, 0, 1], 2, replaceItemUniqueId=line_ids["z"])

    time.sleep(1.0 / 30)
