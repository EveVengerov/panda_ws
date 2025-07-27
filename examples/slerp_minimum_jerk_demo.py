import numpy as np
import pybullet as p
import pybullet_data
import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.trajectory_generator import generate_minimum_jerk_and_slerp_trajectory
from lib.ik import calculate_ik_for_waypoints


# Example usage/demo
if __name__ == "__main__":
    # Connect to PyBullet with GUI
    p.connect(p.GUI)
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Load plane and robot
    p.loadURDF("plane.urdf")
    robot_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
    ee_index = 11  # Panda end effector link

    
    # — Home pose —
    home = [0, -np.pi/4, 0, -np.pi/2, 0, np.pi/3, 0]
        
    for i, q in enumerate(home):
        p.resetJointState(robot_id, i, q)
    time.sleep(1)

    home_position = p.getLinkState(robot_id, ee_index)[0]
    home_orientation = p.getLinkState(robot_id, ee_index)[1]

    # Define waypoints and orientations (quaternions)
    waypoints = [
        home_position, 
        [0.6, 0.3, 0.5],
        [0.6, 0.1, 0.5],
        [0.6, -0.3, 0.5]
    ]
    orientations = [
        home_orientation,
        p.getQuaternionFromEuler([0, np.pi/2, 0]),
        p.getQuaternionFromEuler([0, np.pi/2, 0]),
        p.getQuaternionFromEuler([0, 0, 0])
    ]
    duration = 10.0
    dt = 0.01

    # Generate trajectory
    target_positions, target_orientations = generate_minimum_jerk_and_slerp_trajectory(waypoints, orientations, duration, dt)

    # Get joint limits and rest poses
    n_joints = 7
    lower = []
    upper = []
    ranges = []
    rest = []
    
    for i in range(n_joints):
        joint_info = p.getJointInfo(robot_id, i)
        lower_limit = joint_info[8]
        upper_limit = joint_info[9]
        
        # Handle unlimited joints
        if lower_limit == -1 and upper_limit == -1:
            lower_limit = -np.pi
            upper_limit = np.pi
        
        lower.append(lower_limit)
        upper.append(upper_limit)
        ranges.append(upper_limit - lower_limit)
        
        # Get current joint position as rest pose
        joint_state = p.getJointState(robot_id, i)
        rest.append(joint_state[0])
    
    print(f"Joint limits - Lower: {lower}")
    print(f"Joint limits - Upper: {upper}")
    print(f"Rest poses: {rest}")

    
    joint_solutions = []
    # Compute IK for each pose
    for target_position, target_orientation in zip(target_positions, target_orientations):
        joint_solution = p.calculateInverseKinematics(
            robot_id, ee_index, target_position, target_orientations,
            lowerLimits=lower, upperLimits=upper,
            jointRanges=ranges, restPoses=rest
        )
        joint_solutions.append(joint_solution)

    # Execute trajectory
    print("Executing trajectory...")
    n_joints = 7  # Panda has 7 joints
    prev_ee_pos = None
    prev_traj_pos = None
    for i, joint_sol in enumerate(joint_solutions):
        # Set robot to IK solution using joint motor control
        p.setJointMotorControlArray(
            robot_id,
            range(n_joints),
            p.POSITION_CONTROL,
            targetPositions=joint_sol[:n_joints],
            forces=[87]*n_joints
        )
        
        p.stepSimulation()
        
        # Get current end-effector position
        ee_state = p.getLinkState(robot_id, ee_index)
        curr_ee_pos = ee_state[0]

        # Get current Trajectory position
        curr_traj_pos = target_positions[i]


        # Draw line from previous to current EE position (actual robot path)
        if prev_ee_pos is not None:
            p.addUserDebugLine(prev_ee_pos, curr_ee_pos, [1, 0, 0], 1, lifeTime=0)
        
        if prev_traj_pos is not None:
            p.addUserDebugLine(prev_traj_pos, curr_traj_pos, [0, 1, 0], 3, lifeTime=0)

        prev_ee_pos = curr_ee_pos
        prev_traj_pos = curr_traj_pos
        
        import time
        time.sleep(dt)  # Slow down for visualization

    while True:
        p.setJointMotorControlArray(
        robot_id,
        range(n_joints),
        p.POSITION_CONTROL,
        targetPositions=joint_sol[:n_joints],
        forces=[87]*n_joints
        )
        p.stepSimulation()
        time.sleep(dt)  # Slow down for visualization

    print("Trajectory complete!")
    input("Press Enter to exit...")
    p.disconnect() 