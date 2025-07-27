"""
=============================================================
Script Summary: Dual Panda Pick-and-Place with ArUco Tracking
=============================================================

Main Steps:
1. Initialization:
   - Set up PyBullet simulation, load tables, robots, and cube.
   - Initialize robot joint info, home poses, and camera parameters.
   - Set up ArUco marker detector for vision-based feedback.

2. Pick-and-Place Sequence (Robot A):
   - Move above the cube (pre-grasp pose).
   - Open gripper, descend to grasp pose, and attach the cube.
   - Lift the cube, move to the place location, and hold.
   - Use minimum jerk and SLERP trajectories for smooth motion.

3. Camera and ArUco Marker Search (Robot B):
   - Simulate camera on Robot B's end-effector.
   - Continuously capture images and search for ArUco markers.
   - If marker detected, estimate its pose in the camera frame.

4. ArUco Tracking (Robot B):
   - Transform marker pose from camera frame to robot base frame.
   - Update Robot B's target position to follow the marker in real time.
   - Visualize camera view and marker detection.

5. Lissajous Trajectory (Robot A):
   - After placing the cube in front of the xz plane facing Robot B's camera, Robot A follows a Lissajous curve for demonstration.
   - During this phase, the script computes and optionally plots the tracking error between Robot B and Robot A end-effector positions in the xz plane. Set `track_error_plot = True` to enable error plotting.

Notes:
- The simulation window supports interactive visualization.
- Press Ctrl+C in the terminal to exit the simulation cleanly.
=============================================================
"""
import pybullet as p
import pybullet_data
import numpy as np
import time
import cv2

from lib.simulation_utils import setup_pybullet, load_tables, load_robots, load_cube
from lib.trajectory_generator import MinimumJerkTrajectoryGenerator, SlerpOrientationTrajectoryGenerator, LissajousTrajectoryGenerator
from lib.marker_detector import ArUcoDetector
from lib.robot_utils import open_gripper, close_gripper, get_joint_angles, compute_ik, set_joint_motor_control, get_end_effector_pose,  get_robot_joint_info, set_home_pose
from lib.transform_utils import pose_to_homogeneous, get_T_ee_camera


import signal
import matplotlib.pyplot as plt

# --- Tracking Error Functionality ---
track_error_plot = True  # Set to True to enable error plotting
tracking_errors = []

# Ensure KeyboardInterrupt (Ctrl+C) works
signal.signal(signal.SIGINT, signal.default_int_handler)

# Globals
previous_time = time.time()
current_time = time.time()
is_marker_detected = False
target_pos_camera = None

# compute the error between Robot A and Robot B end-effector positions in the xz plane
def compute_tracking_error(robotA, robotB, ee, plot_flag=False):
    """
    Compute and optionally plot the tracking error between Robot B and Robot A end-effector positions in the xz plane.
    Args:
        robotA: PyBullet ID of Robot A
        robotB: PyBullet ID of Robot B
        ee: End-effector link index
        plot_flag: If True, plot the error over time
    Returns:
        error: Euclidean distance in xz plane
    """
    ee_pos_A = np.array(p.getLinkState(robotA, ee)[0])
    ee_pos_B = np.array(p.getLinkState(robotB, ee)[0])
    error = np.linalg.norm(ee_pos_A[[0,2]] - ee_pos_B[[0,2]])
    if plot_flag:
        tracking_errors.append(error)
        plt.clf()
        plt.title('Tracking Error (Robot B vs Robot A) in XZ Plane')
        plt.xlabel('Timestep')
        plt.ylabel('Error (meters)')
        plt.plot(tracking_errors, label='XZ Error')
        plt.legend()
        plt.pause(0.001)
    return error

# Trajectory Excecutor for Robot A and Robot B during pick-and-place
def move_ee_with_trajectory(start, end, duration, dt, prev_target_ee, prev_pos, orientation):
    """Move the end-effector of the robotA using a minimum jerk trajectory from start to end. Simultaneously move robotB to a predefined position."""
    global previous_time, is_marker_detected, target_pos_camera
    steps = int(duration / dt)
    mini_jerk_traj = MinimumJerkTrajectoryGenerator(start, end, duration, steps)
    for k in range(steps + 50):
        current_time = time.time()
        target_pos_ee, target_vel_ee = mini_jerk_traj.get_target_position(k)
        target_angles = compute_ik(robotA, ee, target_pos_ee, orientation, lower, upper, ranges, rest, n)
        set_joint_motor_control(robotA, n, target_angles)

        # --- Move RobotB to a predefined position to see the plane of RobotA ---
        target_pos_b = [0.5, 0.2, table_offset + 0.5]
        target_orn_b = p.getQuaternionFromEuler([np.pi/2, np.pi/2, 0])
        print(f"[RobotB] Moving to see plane of RobotA {target_pos_b}")
        target_joints_b = compute_ik(robotB, ee, target_pos_b, target_orn_b, lower, upper, ranges, rest, n)
        set_joint_motor_control(robotB, n, target_joints_b)
        # --- ---

        # Add debug lines for visualization: Robot A
        curr_pos = np.array(p.getLinkState(robotA, ee)[0])
        p.addUserDebugLine(prev_pos, curr_pos, [0, 1, 0], 2, lifeTime=5.0)
        prev_pos = curr_pos
        prev_target_ee = target_pos_ee
        show_camera_from_robot_b()  # Show camera view from Robot B

        # Forward the simulation
        p.stepSimulation()
        p.setTimeStep(dt)

        previous_time = current_time
    return prev_target_ee, prev_pos


# Get aruco marker pose from camera and update target position for Robot B in base frame
def follow_camera(robot_b, ee_link_index, rvec, tvec):
    """
    Transform the detected ArUco marker pose from the camera frame to the robot base frame,
    and update the global target_pos_camera variable for robot B to follow.

    Args:
        robot_b: The robot B PyBullet ID.
        ee_link_index: The end-effector link index.
        rvec: Rotation vector of the marker (from ArUco detection).
        tvec: Translation vector of the marker (from ArUco detection).
    """
    global target_pos_camera

    # Get the end-effector pose of robot B (position, orientation, rotation matrix)
    ee_pos, ee_ori, ee_rot = get_end_effector_pose(robot_b, ee_link_index)

    # Compute the homogeneous transformation from robot base to end-effector
    T_base_ee = pose_to_homogeneous(ee_pos, ee_ori)

    # Marker position in camera frame (homogeneous coordinates)
    point_in_ee = np.append(tvec.flatten(), 1.0)

    # Transformation from end-effector to camera frame
    T_ee_camera = get_T_ee_camera()

    # Transform marker position to robot base frame
    point_in_base = T_base_ee @ T_ee_camera @ point_in_ee
    target_pos_camera = point_in_base[:3]

    # Optionally, adjust the y-coordinate to keep robot B at a safe distance from the marker
    target_pos_camera[1] = table_position_b[1] - 0.3

# Attach the camera to Robot B's end-effector 
def show_camera_from_robot_b():
    # Get end-effector pose of robot B
    end_effector_position, end_effector_orientation, end_effector_rotation = get_end_effector_pose(robotB, ee)
    # Show end effector frame axes
    axis_len = 0.2
    p.addUserDebugLine(end_effector_position, end_effector_position + axis_len * end_effector_rotation[:, 0], [1, 0, 0], 2, lifeTime=50*dt)
    p.addUserDebugLine(end_effector_position, end_effector_position + axis_len * end_effector_rotation[:, 1], [0, 1, 0], 2, lifeTime=50*dt)
    p.addUserDebugLine(end_effector_position, end_effector_position + axis_len * end_effector_rotation[:, 2], [0, 0, 1], 2, lifeTime=50*dt)
    
    # Camera frame tarnsform from end-effector
    camera_eye = end_effector_position
    camera_target = end_effector_position + end_effector_rotation[:, 2]
    camera_up = -end_effector_rotation[:, 0] # Use the negative x-axis as up vector for better visualization
    # print(f"end effector rotation: {end_effector_rotation}")
    # print(f"Camera Eye: {camera_eye}, Target: {camera_target}, Up: {camera_up}")

    view_matrix = p.computeViewMatrix(camera_eye, camera_target, camera_up)
    proj_matrix = p.computeProjectionMatrixFOV(fov=fov_deg, aspect=1.0, nearVal=0.01, farVal=2.0)
    img_arr = p.getCameraImage(width, height, viewMatrix=view_matrix, projectionMatrix=proj_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    rgb = np.reshape(img_arr[2], (height, width, 4))
    rgb = rgb[:, :, :3]
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)

    # --- ArUco marker detection ---
    corners, ids = aruco_detector.detect_markers(rgb)
    if ids is not None:
        aruco_poses = aruco_detector.estimate_pose(corners, ids)
        for i, (marker_id, rvec, tvec) in enumerate(aruco_poses):
            global is_marker_detected
            is_marker_detected = True
            rgb = aruco_detector.draw_axes(rgb, rvec, tvec)
            follow_camera(robotB, ee, rvec, tvec)

            # flip image upside down for visualization
            cv2.putText(rgb, f"ID:{marker_id}", (int(aruco_detector.marker_length*100)+10, 30+30*i), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            # print(f"Detected ArUco ID: {marker_id}")
            # print(f"Rotation Vector: {rvec.flatten()}")
            # print(f"Translation Vector: {tvec.flatten()}")

    # Draw crosshair in the middle of the camera image
    center_x = width // 2
    center_y = height // 2
    crosshair_length = 20
    color = (0, 255, 255)
    thickness = 2
    cv2.line(rgb, (center_x - crosshair_length, center_y), (center_x + crosshair_length, center_y), color, thickness)
    cv2.line(rgb, (center_x, center_y - crosshair_length), (center_x, center_y + crosshair_length), color, thickness)
    cv2.imshow('Robot B Camera', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)

def setup_camera(robotB, ee, width=256, height=256, fov_deg=60):
    ee_state_b = p.getLinkState(robotB, ee, computeForwardKinematics=True)
    ee_pos_b = np.array(ee_state_b[4])
    ee_orn_b = np.array(ee_state_b[5])
    ee_rot_b = np.array(p.getMatrixFromQuaternion(ee_orn_b)).reshape(3, 3)
    camera_eye_b = ee_pos_b
    camera_target_b = ee_pos_b + 0.2 * ee_rot_b[:, 0]
    fov_rad = np.deg2rad(fov_deg)
    fy = 0.5 * height / np.tan(fov_rad / 2)
    fx = fy
    cx = width / 2.0
    cy = height / 2.0
    camera_matrix = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ])
    return camera_eye_b, camera_target_b, camera_matrix

def setup_aruco_detector(marker_length, camera_matrix):
    dist_coeffs = np.zeros((5,))
    return ArUcoDetector(marker_length, camera_matrix, dist_coeffs)

if __name__ == "__main__":

    try:
        # Setup PyBullet
        setup_pybullet()

        # Load tables
        table_position_a, table_position_b, tableA, tableB, table_offset = load_tables()

        # Load robots
        robotA, robotB = load_robots(table_position_a, table_position_b, table_offset)

        # Robot joint info
        n, ee, lower, upper, ranges, rest = get_robot_joint_info(robotA)
        target_orn_b = p.getQuaternionFromEuler([np.pi/2, np.pi/2, 0])
        home_b = [0, -np.pi/4, 0, -np.pi/2, 0, np.pi/3, 0]
        set_home_pose(robotB, home_b)

        # Camera setup
        width, height = 256, 256
        fov_deg = 60
        camera_eye_b, camera_target_b, camera_matrix = setup_camera(robotB, ee, width, height, fov_deg)
        marker_length = 0.05*0.8
        aruco_detector = setup_aruco_detector(marker_length, camera_matrix)

        # Set home pose for robotA
        home = [0, -np.pi/4, 0, -np.pi/2, 0, np.pi/3, 0]
        set_home_pose(robotA, home)

        time.sleep(1)
        dt = 0.13

        # Setup cube parameters
        scale = 0.05
        cube_size_meters = 0.05 # size of the cube in meters
        scale_to_meters = cube_size_meters/scale
        cube_start = [0.8, -0.6, cube_size_meters/2 + table_offset]
        place_pose = [0.6, -0.3, cube_size_meters/2 + table_offset + 0.5]
        place_orientation = p.getQuaternionFromEuler([-np.pi/2, np.pi/2, 0])

        # Load cube
        cube_id = load_cube(cube_start, scale_to_meters)

        # get home pose of robotA        
        home_ee_pos = np.array(p.getLinkState(robotA, ee)[0])
        home_orientation = p.getLinkState(robotA, ee)[1]

        # Define positions for the pick and place
        above_cube = [cube_start[0], cube_start[1], cube_start[2] + 0.15]
        above_place = [place_pose[0], place_pose[1], place_pose[2] + 0.15]


        # ----------- PICK AND PLACE SEQUENCE -----------

        # --- Step 1: Move above the cube (pre-grasp pose) ---
        print("[RobotA] Moving above the cube (pre-grasp pose)")
        orientation = p.getQuaternionFromEuler([0, -np.pi, 0])
        prev_target_ee = home_ee_pos
        prev_pos = home_ee_pos
        prev_target_ee, prev_pos = move_ee_with_trajectory(
            start=home_ee_pos,
            end=above_cube,
            duration=3,
            dt=dt,
            prev_target_ee=prev_target_ee,
            prev_pos=prev_pos,
            orientation=orientation
        )

        # --- Step 2: Open gripper before descending to grasp ---
        print("[RobotA] Opening gripper before grasp")
        open_gripper(robotA)

        # --- Step 3: Move down to the cube (grasp pose) ---
        print("[RobotA] Moving down to the cube (grasp pose)")
        prev_target_ee, prev_pos = move_ee_with_trajectory(
            start=prev_pos,
            end=cube_start,
            duration=0.5,
            dt=dt,
            prev_target_ee=prev_target_ee,
            prev_pos=prev_pos,
            orientation=orientation
        )

        # --- Step 4: Attach the cube to the robot's end-effector (simulate grasp) ---
        print("[RobotA] Attaching the cube (simulated grasp)")
        cube_state = p.getLinkState(robotA, ee, computeForwardKinematics=True)
        ee_pos = cube_state[0]
        ee_orn = cube_state[1]
        cube_constraint_id = p.createConstraint(
            parentBodyUniqueId=robotA,
            parentLinkIndex=ee,
            childBodyUniqueId=cube_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
            parentFrameOrientation=ee_orn,
            childFrameOrientation=[0, 0, 0, 1]
        )

        # --- Step 5: Move back up above the cube (post-grasp pose) ---
        print("[RobotA] Moving back up above the cube (post-grasp pose)")
        prev_target_ee, prev_pos = move_ee_with_trajectory(
            start=cube_start,
            end=above_cube,
            duration=1,
            dt=dt,
            prev_target_ee=prev_target_ee,
            prev_pos=prev_pos,
            orientation=orientation
        )

        # --- Step 6: Move to the place location (carry the cube to target) ---
        print("[RobotA] Moving to the place location (carrying the cube)")
        close_gripper(robotA)  # Optionally close gripper to simulate holding
        prev_target_ee, prev_pos = move_ee_with_trajectory(
            start=above_cube,
            end=place_pose,
            duration=5.0,
            dt=dt,
            prev_target_ee=prev_target_ee,
            prev_pos=prev_pos,
            orientation=orientation
        )

        # ----------- SLERP ORIENTATION TRACKING AT PLACE LOCATION -----------
        print("[RobotA] Starting SLERP orientation tracking at place location")
        duration = 3.0
        steps = int(duration/dt)
        curr_orientation = p.getLinkState(robotA, ee)[1]
        traj = SlerpOrientationTrajectoryGenerator(curr_orientation, place_orientation, duration=duration, steps=steps)

        for k in range(steps):
            target_orientation = traj.get_target_orientation(k)
            target_pos_ee = place_pose
            target_angles = compute_ik(robotA, ee, target_pos_ee, target_orientation, lower, upper, ranges, rest, n)
            set_joint_motor_control(robotA, n, target_angles)
            if target_pos_camera is not None:
                print(f"[RobotB] Tracking: Detected marker at {target_pos_camera}")
                target_joints_b = compute_ik(robotB, ee, target_pos_camera, target_orn_b, lower, upper, ranges, rest, n)
                set_joint_motor_control(robotB, n, target_joints_b)
            p.addUserDebugLine(prev_target_ee, target_pos_ee, [1, 0, 0], 1, lifeTime=0)
            curr_pos = np.array(p.getLinkState(robotA, ee)[0])
            p.addUserDebugLine(prev_pos, curr_pos, [0, 1, 0], 2, lifeTime=0)
            prev_pos = curr_pos
            prev_target_ee = target_pos_ee
            show_camera_from_robot_b()
            p.stepSimulation()
            p.setTimeStep(dt)
            time.sleep(dt)

        # ----------- TRANSITION TO LISSAJOUS TRAJECTORY TRACKING -----------
        print("[RobotA] Transitioning to Lissajous trajectory tracking")
        # Trajectory parameters - Lissajous Trajectory
        A, B = 0.15, 0.15
        a = 1
        b = 2
        delta = np.pi / 32
        place_pose[2] += 0.2
        center = np.array(place_pose)
        g = np.gcd(a, b)
        T = 2*np.pi/g
        dt = 0.02
        steps = int(T/dt) + 1

        lissajous_trajectory = LissajousTrajectoryGenerator(A, B, a, b, delta, center, T, steps, plane='xz')
        lissajous_trajectory.get_plot()

        k = 0
        previous_time = time.time()

        print("✅ Pick and place complete. Starting Lissajous trajectory tracking. Press Ctrl+C to exit.")
        while True:
            # --- Lissajous trajectory tracking loop ---
            current_time = time.time()
            target = lissajous_trajectory.get_target_position(k)
            ori = p.getQuaternionFromEuler([0, np.pi/2, 0])

            angles = compute_ik(robotA, ee, target, place_orientation, lower, upper, ranges, rest, n)

            set_joint_motor_control(robotA, n, angles, forces=[90]*n, velocities=[0]*n)

            show_camera_from_robot_b()
            if target_pos_camera is not None:
                print(f"[RobotB] Tracking: Detected marker at {target_pos_camera}")
                target_joints_b = compute_ik(robotB, ee, target_pos_camera, target_orn_b, lower, upper, ranges, rest, n)
                set_joint_motor_control(robotB, n, target_joints_b)

            # --- Compute and (optionally) plot tracking error in XZ plane ---
            error = compute_tracking_error(robotA, robotB, ee, plot_flag=track_error_plot)
            if not track_error_plot:
                print(f"Tracking error (XZ plane): {error:.4f} m")

            p.stepSimulation()
            p.setTimeStep(dt)

            curr_pos = np.array(p.getLinkState(robotA, ee)[0])
            p.addUserDebugLine(prev_pos, curr_pos, [1, 0, 0], 2, lifeTime=0)
            prev_pos = curr_pos

            k = k + 1
    except KeyboardInterrupt:
        print("\n⏹ Ctrl+C received — exiting.")
    finally:
        p.disconnect()
        cv2.destroyAllWindows()