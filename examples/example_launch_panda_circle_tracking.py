"""
=============================================================
Script Summary: Panda Simple Trajectory Demo Example to test out simple trajectory tracking
=============================================================
This script demonstrates a basic trajectory following task for the Franka Panda robot in PyBullet.

Main Features:
- Loads a single Panda robot in a simulated environment.
- Sets the robot to a home pose.
- Generates a circular trajectory for the end-effector in the YZ plane.
- Uses inverse kinematics (IK) to compute joint angles for each target point.
- Visualizes the end-effector path with debug lines in the simulation.

Usage:
    python launch_panda_demo.py
=============================================================
"""
import numpy as np
import pybullet as p
import pybullet_data
import time

# — Connect and load robot —
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

# — Joint info and link index —
n = 7
ee_link = 11  # end-effector link index for Panda

# — Joint limits setup —
infos = [p.getJointInfo(robot, i) for i in range(p.getNumJoints(robot))]
lower = [info[8] for info in infos[:n]]
upper = [info[9] for info in infos[:n]]
ranges = [upper[i] - lower[i] for i in range(n)]
rest = [0]*n

# — Set home pose —
home = [0, -np.pi/4, 0, -np.pi/2, 0, np.pi/3, 0]
for i, q in enumerate(home):
    p.resetJointState(robot, i, q)

time.sleep(1)

# — Trajectory parameters (circle) —
center = np.array([0.5, 0, 0.5])
radius = 0.1
steps = 400
T = 8.0

# Get initial end-effector position
ee_state = p.getLinkState(robot, ee_link)
prev_pos = np.array(ee_state[0])

# — Main loop: IK + visualization —
for t in range(1, steps+1):
    theta = 2*np.pi * t / steps
    target = center + radius * np.array([0, np.cos(theta), np.sin(theta)])
    ori = p.getQuaternionFromEuler([0, np.pi/2, 0])

    ik = p.calculateInverseKinematics(robot, ee_link, target, ori,
                                      lowerLimits=lower, upperLimits=upper,
                                      jointRanges=ranges, restPoses=rest)
    angles = ik[:n]
    p.setJointMotorControlArray(robot, list(range(n)),
                                p.POSITION_CONTROL, angles, forces=[87]*n)

    p.stepSimulation()
    time.sleep(T/steps)

    # Draw a line from previous EE position to current
    state = p.getLinkState(robot, ee_link)
    curr_pos = np.array(state[0])
    p.addUserDebugLine(prev_pos, curr_pos, [1, 0, 0], lineWidth=2, lifeTime=0)
    prev_pos = curr_pos

print("✅ Trajectory complete")
time.sleep(2)
p.disconnect()
