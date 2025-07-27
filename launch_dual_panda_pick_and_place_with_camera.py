"""
=============================================================
Script Summary: Dual Panda Pick-and-Place with Camera Demo
=============================================================
This script demonstrates a dual Franka Panda robot setup in PyBullet, performing a pick-and-place task with a camera pointing at the workspace.

Main Features:
- Loads two Panda robots and two tables in a simulated environment.
- Robot A performs a pick-and-place task using minimum jerk and SLERP trajectories.
- Robot B is positioned to observe the workspace with a simulated camera attached to its end-effector.
- Real-time camera visualization using OpenCV.

Usage:
    python launch_dual_panda_pick_and_place_with_camera.py
=============================================================
"""
import pybullet as p
import pybullet_data
import numpy as np
import time
import cv2
from lib.trajectory_generator import MinimumJerkTrajectoryGenerator, SlerpOrientationTrajectoryGenerator
import signal
import matplotlib.pyplot as plt  # <-- Added for plotting

# Ensure KeyboardInterrupt (Ctrl+C) works
signal.signal(signal.SIGINT, signal.default_int_handler)

def open_gripper(robot_id):
    p.setJointMotorControl2(robot_id, 9, p.POSITION_CONTROL, targetPosition=0.04, force=20)
    p.setJointMotorControl2(robot_id, 10, p.POSITION_CONTROL, targetPosition=0.04, force=20)

def close_gripper(robot_id):
    p.setJointMotorControl2(robot_id, 9, p.POSITION_CONTROL, targetPosition=0.0, force=25)
    p.setJointMotorControl2(robot_id, 10, p.POSITION_CONTROL, targetPosition=0.0, force=25)

def get_joint_angles(robot_id, joint_indices):
    return [p.getJointState(robot_id, i)[0] for i in joint_indices]

def follow_trajectory(prev_target_ee, home_ee_pos, robot, ee, orientation, lower, upper, ranges, rest, mini_jerk_traj, steps, dt):
    prev_target_ee = home_ee_pos
    prev_pos = home_ee_pos
    errors = []
    plt.ion()
    line, = ax.plot([], [], 'r-')
    ax.set_xlabel('Step')
    ax.set_ylabel('Tracking Error (m)')
    ax.set_title('End-Effector Tracking Error')
    ax.set_ylim(0, 0.2)
    ax.set_xlim(0, steps+2)
    for k in range(steps + 50):
        target_pos_ee, target_vel_ee  = mini_jerk_traj.get_target_position(k)
        target_angles = p.calculateInverseKinematics(robot, ee, target_pos_ee, orientation,
                                                  lowerLimits=lower, upperLimits=upper,
                                                  jointRanges=ranges, restPoses=rest,
                                                  residualThreshold=1e-3)[:n]

        curr_pos = np.array(p.getLinkState(robot, ee)[0])
        error_position = np.linalg.norm(curr_pos - target_pos_ee)
        errors.append(error_position)

        p.setJointMotorControlArray(robot, range(n),
                                    p.POSITION_CONTROL,
                                    targetPositions=target_angles,
                                    forces=[87]*n)

        p.addUserDebugLine(prev_target_ee, target_pos_ee, [1, 0, 0], 1, lifeTime=0)
        curr_pos = np.array(p.getLinkState(robot, ee)[0])
        p.addUserDebugLine(prev_pos, curr_pos, [0, 1, 0], 2, lifeTime=0)
        prev_pos = curr_pos
        prev_target_ee = target_pos_ee

        p.stepSimulation()
        p.setTimeStep(dt)
        # time.sleep(dt)



        # Update plot
        line.set_xdata(np.arange(len(errors)))
        line.set_ydata(errors)
        ax.set_xlim(0, max(steps+2, len(errors)))
        ax.set_ylim(0, max(0.2, max(errors)+0.01))
        plt.pause(dt)

        show_camera_from_robot_b()    


    # plt.ioff()
    # plt.show(block=False)
    return prev_target_ee, prev_pos

def show_camera_from_robot_b():
    # Get end-effector pose of robot B
    state = p.getLinkState(robotB, ee, computeForwardKinematics=True)
    pos = np.array(state[4])
    orn = np.array(state[5])
    rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
    # Show end effector frame axes
    axis_len = 0.2
    p.addUserDebugLine(pos, pos + axis_len * rot[:, 0], [1, 0, 0], 2, lifeTime=50*dt)  # x-axis (red)
    p.addUserDebugLine(pos, pos + axis_len * rot[:, 1], [0, 1, 0], 2, lifeTime=50*dt)  # y-axis (green)
    p.addUserDebugLine(pos, pos + axis_len * rot[:, 2], [0, 0, 1], 2, lifeTime=50*dt)  # z-axis (blue)
    # Camera parameters (rotated 90 deg around EE y-axis)
    camera_eye = pos
    camera_target = pos + 0.2 * rot[:, 2]  # EE z-axis (was x-axis)
    camera_up = -rot[:, 0]  # -EE x-axis (was z-axis)
    view_matrix = p.computeViewMatrix(camera_eye, camera_target, camera_up)
    proj_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=1.0, nearVal=0.01, farVal=2.0)
    width, height = 256, 256
    img_arr = p.getCameraImage(width, height, viewMatrix=view_matrix, projectionMatrix=proj_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    rgb = np.reshape(img_arr[2], (height, width, 4))
    rgb = rgb[:, :, :3]
    cv2.imshow('Robot B Camera', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)

try:
    # Setup PyBullet
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    # Load plane
    p.loadURDF("plane.urdf")

    # Load tables
    table_position_a = [0.5, -0.6, 0]
    table_position_b = [0.5,  0.6, 0]
    tableA = p.loadURDF("table/table.urdf", basePosition=table_position_a)
    tableB = p.loadURDF("table/table.urdf", basePosition=table_position_b)
    table_offset = 0.62

    # Load robots
    robotA = p.loadURDF("franka_panda/panda.urdf", basePosition=[table_position_a[0], table_position_a[1], table_offset], useFixedBase=True)
    robotB = p.loadURDF("franka_panda/panda.urdf", basePosition=[table_position_b[0], table_position_b[1], table_offset], useFixedBase=True)
    
    n = 7
    ee = 11  # End-effector link index
    infos = [p.getJointInfo(robotA, i) for i in range(n)]
    lower = [info[8] for info in infos]
    upper = [info[9] for info in infos]
    ranges = [upper[i] - lower[i] for i in range(n)]
    rest = [0]*n

    # Set robot B end effector to face -y axis toward xz plane
    target_pos_b = [0.5, 0.3, table_offset + 0.3]  # Example position above table B
    target_orn_b = p.getQuaternionFromEuler([np.pi/2, np.pi/2, 0])  # Rotate EE x-axis to -y (world)
    target_joints_b = p.calculateInverseKinematics(robotB, ee, target_pos_b, target_orn_b)[:n]
    for i in range(n):
        p.resetJointState(robotB, i, target_joints_b[i])
    # Add debug line for camera direction (EE x-axis)
    ee_state_b = p.getLinkState(robotB, ee, computeForwardKinematics=True)
    ee_pos_b = np.array(ee_state_b[4])
    ee_orn_b = np.array(ee_state_b[5])
    ee_rot_b = np.array(p.getMatrixFromQuaternion(ee_orn_b)).reshape(3, 3)
    camera_eye_b = ee_pos_b
    camera_target_b = ee_pos_b + 0.2 * ee_rot_b[:, 0]  # x-axis of EE

    # Home pose for robot A
    home = [0, -np.pi/4, 0, -np.pi/2, 0, np.pi/3, 0]
    for i, q in enumerate(home):
        p.resetJointState(robotA, i, q)
    time.sleep(1)
    dt = 0.05

    # cube_id = p.loadURDF("cube_small.urdf", cube_start, globalScaling=1.0)
    # The mesh in my URDF is scaled 0.05, so to use globalScaling ...
    scale = 0.05
    cube_size_meters = 0.05  # Size of the cube in meters
    scale_to_meters = cube_size_meters/scale

    # Add cube
    cube_start = [0.8, -0.6, cube_size_meters/2 + table_offset]
    # place_pose = [0.6, -0.9, cube_size_meters/2 + table_offset]
    place_pose = [0.6, -0.3, cube_size_meters/2 + table_offset + 0.5]
    place_orientation = p.getQuaternionFromEuler([-np.pi/2, np.pi/2, 0])

    cube_id = p.loadURDF(
        "aruco_cube_description/urdf/aruco.urdf",
        cube_start,
        globalScaling=scale_to_meters
    )

    texture_id = p.loadTexture("aruco_cube_description/materials/textures/aruco_0_with_border.png")
    p.changeVisualShape(cube_id, -1, textureUniqueId=texture_id)

    # Get home end-effector position using FK
    home_ee_pos = np.array(p.getLinkState(robotA, ee)[0])
    home_orientation = p.getLinkState(robotA, ee)[1]
    # Target poses
    above_cube = [cube_start[0], cube_start[1], cube_start[2] + 0.15]
    above_place = [place_pose[0], place_pose[1], place_pose[2] + 0.15]
    orientation = p.getQuaternionFromEuler([0, -np.pi, 0])
    # Step 1: Go from Home to above cube position
    mini_jerk_traj = MinimumJerkTrajectoryGenerator(
        start=home_ee_pos,
        end=above_cube,
        duration=3,
        dt=dt
    )
    steps = mini_jerk_traj.steps
    prev_target_ee = home_ee_pos
    prev_pos = home_ee_pos
    fig, ax = plt.subplots()

    prev_target_ee, prev_pos = follow_trajectory(prev_target_ee, home_ee_pos, robotA, ee, orientation, lower, upper,
                                                  ranges, rest, mini_jerk_traj, steps, dt)

    #clear plot
    ax.clear()


    open_gripper(robotA)
    # Step 3: Move down to pick cube
    mini_jerk_traj = MinimumJerkTrajectoryGenerator(
        start=prev_pos,
        end=cube_start,
        duration=10,
        dt=dt
    )
    steps = mini_jerk_traj.steps

    prev_target_ee, prev_pos = follow_trajectory(prev_target_ee, prev_pos, robotA, ee, orientation, lower, upper,
                                                  ranges, rest, mini_jerk_traj, steps, dt)
    close_gripper(robotA)

    ax.clear()

    # Step 5: Move Cube to above cube position
    mini_jerk_traj = MinimumJerkTrajectoryGenerator(
        start=cube_start,
        end=above_cube,
        duration=1,
        dt=dt
    )
    steps = mini_jerk_traj.steps

    prev_target_ee, prev_pos = follow_trajectory(prev_target_ee, prev_pos, robotA, ee, orientation, lower, upper,
                                                  ranges, rest, mini_jerk_traj, steps, dt)

    # Step 6: Move to  place position
    mini_jerk_traj = MinimumJerkTrajectoryGenerator(
        start=above_cube,
        end=place_pose,
        duration=10.0,
        dt=dt
    )
    steps = mini_jerk_traj.steps

    ax.clear()

    prev_target_ee, prev_pos = follow_trajectory(prev_target_ee, prev_pos, robotA, ee, orientation, lower, upper,
                                                  ranges, rest, mini_jerk_traj, steps, dt)

    steps = int(1/dt)
    curr_orientation = p.getLinkState(robotA, ee)[1]
    traj = SlerpOrientationTrajectoryGenerator(curr_orientation, place_orientation, duration=1.0, steps=steps)
    curr_joint_angles = get_joint_angles(robotA, range(n))

    # Optional: Plot error for orientation trajectory as well
    errors = []
    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [], 'b-')
    ax.set_xlabel('Step')
    ax.set_ylabel('Tracking Error (m)')
    ax.set_title('End-Effector Tracking Error (Orientation Slerp)')
    ax.set_ylim(0, 0.2)
    ax.set_xlim(0, steps)
    prev_target_ee = prev_pos
    prev_pos = prev_pos
    for k in range(steps):
        target_orientation = traj.get_target_orientation(k)
        # target_pos_ee, target_vel_ee  = mini_jerk_traj.get_target_position(k)
        target_pos_ee = place_pose
        target_angles = p.calculateInverseKinematics(robotA, ee, target_pos_ee, target_orientation,
                                                  lowerLimits=lower, upperLimits=upper,
                                                  jointRanges=ranges, restPoses=curr_joint_angles)[:n]
        p.setJointMotorControlArray(robotA, range(n),
                                    p.POSITION_CONTROL,
                                    targetPositions=target_angles,
                                    forces=[87]*n)
        p.addUserDebugLine(prev_target_ee, target_pos_ee, [1, 0, 0], 1, lifeTime=0)
        curr_pos = np.array(p.getLinkState(robotA, ee)[0])
        p.addUserDebugLine(prev_pos, curr_pos, [0, 1, 0], 2, lifeTime=0)
        prev_pos = curr_pos
        prev_target_ee = target_pos_ee
        # Camera output from robot B
        show_camera_from_robot_b()
        p.stepSimulation()
        p.setTimeStep(dt)
        time.sleep(dt)
        # Error calculation and plot
        error_position = np.linalg.norm(curr_pos - target_pos_ee)
        errors.append(error_position)
        line.set_xdata(np.arange(len(errors)))
        line.set_ydata(errors)
        ax.set_xlim(0, max(steps, len(errors)))
        ax.set_ylim(0, max(0.2, max(errors)+0.01))
        plt.pause(0.001)
    plt.ioff()
    plt.show(block=False)

    print("✅ Pick and place complete. Press Ctrl+C to exit.")
    while True:
        show_camera_from_robot_b()
        p.stepSimulation()
        time.sleep(0.01)
except KeyboardInterrupt:
    print("\n⏹ Ctrl+C received — exiting.")
finally:

    p.disconnect()
    cv2.destroyAllWindows()
    print("✅ Disconnected from PyBullet.")