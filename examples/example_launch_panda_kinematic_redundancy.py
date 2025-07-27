"""
=============================================================
Script Summary: Panda Kinematic Redundancy Demo
=============================================================
This script demonstrates the concept of kinematic redundancy for the Franka Panda robot in PyBullet.

Main Features:
- Loads a single Panda robot in a simulated environment.
- Defines multiple null space goals (redundant joint configurations) for the same end-effector pose.
- Moves the robot to each null space goal, then solves IK to reach a fixed end-effector target position/orientation
  while biasing the solution toward the current null space goal.
- Visualizes how the robot can achieve the same end-effector pose with different joint configurations.
- Useful for understanding redundancy resolution and null space control in redundant manipulators.

Usage:
    python launch_panda_kinematic_redundancy.py
=============================================================
"""

import pybullet as p
import pybullet_data
import numpy as np
import time


from lib.robot_utils import set_home_pose

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
ee_link = 11


home = [0, -np.pi/4, 0, -np.pi/2, 0, np.pi/3, 0]
set_home_pose(robot, home)


# get joint limits and ranges
n = 7
infos = [p.getJointInfo(robot, i) for i in range(n)]
lower_limits = [info[8] for info in infos]
upper_limits = [info[9] for info in infos]
joint_ranges = [upper_limits[i] - lower_limits[i] for i in range(n)]

joint_limit_centers = [(upper_limits[i] + lower_limits[i]) / 2 for i in range(n)]


target_pos = [0.5, 0, 0.5]
target_orn = p.getQuaternionFromEuler([0, np.pi/2, 0])

# Different null space goals (redundant configurations)
null_space_goals = [
    home, # Home pose
    joint_limit_centers, # Center of joint limits
    [0.0, -1.0, 0.0, -2.5, 0.0, 2.5, 0.5],   # Elbow back
]

##
labels = ["Home", "Elbow Out", "Elbow Back"] 

n = 7
infos = [p.getJointInfo(robot, i) for i in range(p.getNumJoints(robot))]
lower_limits = [info[8] for info in infos[:n]]
upper_limits = [info[9] for info in infos[:n]]
joint_ranges = [upper_limits[i] - lower_limits[i] for i in range(n)]



for idx, rest in enumerate(null_space_goals):
    # set joint states to the null space goals

    print(f"Setting null space goal {idx+1}: {rest}")
    p.setJointMotorControlArray(
        robot, range(n),
        p.POSITION_CONTROL,
        targetPositions=rest,
        forces=[87]*n
    )

    print(f"Null space goal name : ", labels[idx])

    # Simulate to settle
    for _ in range(600):
        p.stepSimulation()
        time.sleep(1. / 240.)

    print(f"Moving to null space solution {idx+1}")
    target_angles = p.calculateInverseKinematics(
        robot, ee_link, target_pos, target_orn,
        lowerLimits=lower_limits, upperLimits=upper_limits,
        jointRanges=joint_ranges, restPoses=rest, residualThreshold=1e-3
    )[:n]
    

    p.setJointMotorControlArray(
        robot, range(n),
        p.POSITION_CONTROL,
        targetPositions=target_angles,
        forces=[87]*n
    )

    # Simulate to settle
    for _ in range(200):
        p.stepSimulation()
        time.sleep(1. / 240.)

    # Pause to inspect visually
    time.sleep(2)

p.disconnect()
