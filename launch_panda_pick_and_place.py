"""
=============================================================
Script Summary: Panda Pick-and-Place Demo
=============================================================
This script demonstrates a basic pick-and-place task for the Franka Panda robot in PyBullet.

Main Features:
- Loads a single Panda robot and a cube in a simulated environment.
- Uses minimum jerk trajectories for smooth end-effector motion.
- Executes a full pick-and-place sequence:
    1. Move above the cube
    2. Open gripper
    3. Move down to pick
    4. Close gripper (pick)
    5. Move up
    6. Move to place location
    7. Move down to place
    8. Open gripper (place)
    9. Return to home
- Visualizes the end-effector path with debug lines in the simulation.

Usage:
    python launch_panda_pick_and_place.py
=============================================================
"""


import numpy as np
import pybullet as p
import pybullet_data
import time
import signal

from lib.trajectory_generator import MinimumJerkTrajectoryGenerator

# Ensure KeyboardInterrupt (Ctrl+C) works
signal.signal(signal.SIGINT, signal.default_int_handler)

def open_gripper(robot_id):
    p.setJointMotorControl2(robot_id, 9, p.POSITION_CONTROL, targetPosition=0.04, force=20)
    p.setJointMotorControl2(robot_id, 10, p.POSITION_CONTROL, targetPosition=0.04, force=20)

def close_gripper(robot_id):
    p.setJointMotorControl2(robot_id, 9, p.POSITION_CONTROL, targetPosition=0.0, force=20)
    p.setJointMotorControl2(robot_id, 10, p.POSITION_CONTROL, targetPosition=0.0, force=20)


def follow_trajectory(prev_target_ee, home_ee_pos, robot, ee, orientation, lower, upper, ranges, rest, mini_jerk_traj, steps, dt):
    
    prev_target_ee = home_ee_pos
    prev_pos = home_ee_pos

    
    for k in range(steps + 10):
        
        target_pos_ee, target_vel_ee  = mini_jerk_traj.get_target_position(k)
        target_angles = p.calculateInverseKinematics(robot, ee, target_pos_ee, orientation,
                                                  lowerLimits=lower, upperLimits=upper,
                                                  jointRanges=ranges, restPoses=rest)[:n]
        
        p.setJointMotorControlArray(robot, range(n),
                                    p.POSITION_CONTROL,
                                    targetPositions=target_angles,
                                    forces=[87]*n)
        
        # plot target ee in simulation
        p.addUserDebugLine(prev_target_ee, target_pos_ee, [1, 0, 0], 1, lifeTime=0)
        # add current position to simulation
        curr_pos = np.array(p.getLinkState(robot, ee)[0])
        p.addUserDebugLine(prev_pos, curr_pos, [0, 1, 0], 2, lifeTime=0)
        prev_pos = curr_pos
        prev_target_ee = target_pos_ee
        # print(f"Step {k+1}/{steps}: \nTarget EE Position: {target_ee}, \nCurrent Position: {curr_pos}")
        p.stepSimulation()
        p.setTimeStep(dt)
        time.sleep(dt)
        
    return prev_target_ee, prev_pos

try:
    # — Setup PyBullet —
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    plane = p.loadURDF("plane.urdf")
    robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

    # — Joint and IK setup —
    n = 7
    ee = 11  # End-effector link index
    infos = [p.getJointInfo(robot, i) for i in range(n)]
    lower = [info[8] for info in infos]
    upper = [info[9] for info in infos]
    ranges = [upper[i] - lower[i] for i in range(n)]
    rest = [0]*n

    # — Home pose —
    home = [0, -np.pi/4, 0, -np.pi/2, 0, np.pi/3, 0]
        
    for i, q in enumerate(home):
        p.resetJointState(robot, i, q)
    time.sleep(1)
    # Simulation timestep
    dt = 0.01

    # — Add cube —
    cube_start = [0.6, 0, 0.025]
    cube_id = p.loadURDF("cube_small.urdf", cube_start, globalScaling=1.0)

    # Get home end-effector position using FK
    home_ee_pos = np.array(p.getLinkState(robot, ee)[0])
    home_orientation = p.getLinkState(robot, ee)[1]
    print(f"Home end-effector position: {home_ee_pos}")
    
    # — Target pose for pick (above cube) —
    above_cube = [cube_start[0], cube_start[1], cube_start[2] + 0.15]
    place_pose = [0.4, -0.3, 0.025]
    above_place = [place_pose[0], place_pose[1], place_pose[2] + 0.15]
    
    orientation = p.getQuaternionFromEuler([0, -np.pi, 0])
    
    # Step 1: Go from Home to above cube position
    mini_jerk_traj = MinimumJerkTrajectoryGenerator(
        start=home_ee_pos,
        end=above_cube,
        duration=2.0,
        dt=dt
    )
    
    steps = mini_jerk_traj.steps
    print(f"Steps: {steps}, dt: {dt}")

    # — Move to above cube position —
    # Step 1
    prev_target_ee = home_ee_pos
    prev_pos = home_ee_pos
    
    prev_target_ee, prev_pos = follow_trajectory(prev_target_ee, home_ee_pos, robot, ee, orientation, lower, upper,
                                                  ranges, rest, mini_jerk_traj, steps, dt)
    print("✅ Moved to above cube position.")
        
    # Step 2: Open gripper
    open_gripper(robot)
    
    # Step 3: Move down to pick cube
    mini_jerk_traj = MinimumJerkTrajectoryGenerator(
        start=prev_pos,
        end=cube_start,
        duration=1.0,
        dt=dt
    )
    prev_target_ee, prev_pos = follow_trajectory(prev_target_ee, prev_pos, robot, ee, orientation, lower, upper,
                                                  ranges, rest, mini_jerk_traj, steps, dt)  
    print("✅ Moved down to cube position.")
        
    # Step 4: Pick Cube / close gripper
    close_gripper(robot)
    
    # Step 5: Move Cube to above cube position
    mini_jerk_traj = MinimumJerkTrajectoryGenerator(
        start=cube_start,
        end=above_cube,
        duration=1.0,
        dt=dt
    )
    prev_target_ee, prev_pos = follow_trajectory(prev_target_ee, prev_pos, robot, ee, orientation, lower, upper,
                                                  ranges, rest, mini_jerk_traj, steps, dt)
    print("✅ Moved Cube to above cube position.")
        
    # Step 6: Move to above place position
    mini_jerk_traj = MinimumJerkTrajectoryGenerator(
        start=above_cube,
        end=above_place,
        duration=2.0,
        dt=dt
    )
    prev_target_ee, prev_pos = follow_trajectory(prev_target_ee, prev_pos, robot, ee, orientation, lower, upper,
                                                  ranges, rest, mini_jerk_traj, steps, dt)
    print("✅ Moved to above place position.")
    
    # Step 7: Move down to place position
    mini_jerk_traj = MinimumJerkTrajectoryGenerator(
        start=above_place,
        end=place_pose,
        duration=1.0,
        dt=dt
    )
    prev_target_ee, prev_pos = follow_trajectory(prev_target_ee, prev_pos, robot, ee, orientation, lower, upper,
                                                  ranges, rest, mini_jerk_traj, steps, dt)
    print("✅ Moved down to place position.")
    
    # Step 8: Place Cube / Open Gripper
    open_gripper(robot)
    
    # — Move back to above place position —
    mini_jerk_traj = MinimumJerkTrajectoryGenerator(
        start=place_pose,
        end=above_place,
        duration=1.0,
        dt=dt
    )
    prev_target_ee, prev_pos = follow_trajectory(prev_target_ee, prev_pos, robot, ee, orientation, lower, upper,
                                                  ranges, rest, mini_jerk_traj, steps, dt)
    print("✅ Moved back to above place position.")
    
    # — Close gripper —
    close_gripper(robot)
    
    # — Move back to home position —
    mini_jerk_traj = MinimumJerkTrajectoryGenerator(
        start=above_place,
        end=home_ee_pos,
        duration=2.0,
        dt=dt
    )
    prev_target_ee, prev_pos = follow_trajectory(prev_target_ee, prev_pos, robot, ee, orientation, lower, upper,
                                                  ranges, rest, mini_jerk_traj, steps, dt)
    print("✅ Moved back to home position.")
    
    # set orientation to home orientation
    home_orientation = p.getQuaternionFromEuler([0, np.pi, 0])
    target_angles = p.calculateInverseKinematics(robot, ee, home_ee_pos, home_orientation,
                                                lowerLimits=lower, upperLimits=upper,
                                                jointRanges=ranges, restPoses=rest)[:n]
    p.setJointMotorControlArray(robot, range(n),
                                p.POSITION_CONTROL,
                                targetPositions=target_angles,
                                forces=[87]*n)    
    p.stepSimulation()
    p.setTimeStep(dt)
    time.sleep(dt)
        
    print("✅ Pick and place complete. Press Ctrl+C to exit.")
    while True:
        p.stepSimulation()
        time.sleep(0.01)

except KeyboardInterrupt:
    print("\n⏹ Ctrl+C received — exiting.")

finally:
    p.disconnect()
    print("Simulation disconnected.")
