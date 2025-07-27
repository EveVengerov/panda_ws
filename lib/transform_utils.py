import pybullet as p
import numpy as np

def pose_to_homogeneous(position, orientation):
    rot_matrix = p.getMatrixFromQuaternion(orientation)
    rot_matrix = np.array(rot_matrix).reshape(3, 3)
    T = np.eye(4)
    T[:3, :3] = rot_matrix
    T[:3, 3] = position
    return T

def get_T_ee_camera():
    T_ee_camera = np.eye(4)
    # Define the transformation from end-effector to camera frame
    # This is a fixed transformation based on the camera's position relative to the end-effector
    # This means:
    # Camera +x axis ← robot's +y
    # Camera +y axis ← robot's −x
    # Camera +z axis ← robot's +z
    T_ee_camera[:3, :3] = np.array([
        [0, 1, 0],
        [-1,  0, 0],
        [0,  0, 1]
    ])
    T_ee_camera[:3, 3] = [0, 0, 0]
    return T_ee_camera