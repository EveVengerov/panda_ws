import pybullet as p
import numpy as np

def open_gripper(robot_id):
    p.setJointMotorControl2(robot_id, 9, p.POSITION_CONTROL, targetPosition=0.04, force=20)
    p.setJointMotorControl2(robot_id, 10, p.POSITION_CONTROL, targetPosition=0.04, force=20)

def close_gripper(robot_id):
    p.setJointMotorControl2(robot_id, 9, p.POSITION_CONTROL, targetPosition=0.025, force=2)
    p.setJointMotorControl2(robot_id, 0, p.POSITION_CONTROL, targetPosition=0.025, force=2)

def get_joint_angles(robot_id, joint_indices):
    return [p.getJointState(robot_id, i)[0] for i in joint_indices]

def get_end_effector_pose(robot_id, ee_index):
    """Get the end-effector pose (position , orientation and rotation matrix) of the robot."""
    state = p.getLinkState(robot_id, ee_index, computeForwardKinematics=True)
    pos = np.array(state[4])
    orn = np.array(state[5])
    rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
    return pos, orn, rot

def get_robot_joint_info(robot):
    n = 7
    ee = 11
    infos = [p.getJointInfo(robot, i) for i in range(n)]
    lower = [info[8] for info in infos]
    upper = [info[9] for info in infos]
    ranges = [upper[i] - lower[i] for i in range(n)]
    rest = [0]*n
    return n, ee, lower, upper, ranges, rest

def set_home_pose(robot, home=[0, -np.pi/4, 0, -np.pi/2, 0, np.pi/3, 0]):
    for i, q in enumerate(home):
        p.resetJointState(robot, i, q)


def compute_ik(robot, ee, target_pos, target_orn, lower, upper, ranges, rest, n, residualThreshold=1e-3):
    """Compute inverse kinematics for a given robot and end-effector."""
    return p.calculateInverseKinematics(
        robot, ee, target_pos, target_orn,
        lowerLimits=lower, upperLimits=upper,
        jointRanges=ranges, restPoses=rest,
        residualThreshold=residualThreshold
    )[:n]

def set_joint_motor_control(robot, n, target_angles, forces=None, velocities=None):
    """Set joint motor control for a robot."""
    if forces is None:
        forces = [87] * n
    if velocities is not None:
        p.setJointMotorControlArray(
            robot, range(n),
            p.POSITION_CONTROL,
            targetPositions=target_angles,
            targetVelocities=velocities,
            forces=forces
        )
    else:
        p.setJointMotorControlArray(
            robot, range(n),
            p.POSITION_CONTROL,
            targetPositions=target_angles,
            forces=forces
        )
