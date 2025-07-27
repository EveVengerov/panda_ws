import cv2
import cv2.aruco as aruco
import numpy as np

class ArUcoDetector:
    def __init__(self, marker_length, camera_matrix, dist_coeffs, aruco_dict_type=aruco.DICT_4X4_100):
        self.marker_length = marker_length
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.aruco_dict = aruco.getPredefinedDictionary(aruco_dict_type)

    def detect_markers(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict)
        return corners, ids

    def estimate_pose(self, corners, ids):
        poses = []
        half_len = self.marker_length / 2.0

        # 3D coordinates of the marker corners in marker's local coordinate frame
        obj_points = np.array([
            [-half_len,  half_len, 0],
            [ half_len,  half_len, 0],
            [ half_len, -half_len, 0],
            [-half_len, -half_len, 0]
        ], dtype=np.float32)

        if ids is not None:
            for i, marker_id in enumerate(ids):
                img_points = corners[i][0].astype(np.float32)
                success, rvec, tvec = cv2.solvePnP(
                    obj_points, img_points, 
                    self.camera_matrix, self.dist_coeffs, 
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                if success:
                    poses.append((marker_id[0], rvec, tvec))

        return poses

    def draw_axes(self, frame, rvec, tvec):
        axis_length = self.marker_length * 2.0
        axis_points = np.float32([
            [0, 0, 0],                # origin
            [axis_length, 0, 0],      # X axis
            [0, axis_length, 0],      # Y axis
            [0, 0, axis_length]       # Z axis
        ])

        imgpts, _ = cv2.projectPoints(axis_points, rvec, tvec, self.camera_matrix, self.dist_coeffs)
        imgpts = imgpts.astype(int).reshape(-1, 2)

        origin = tuple(imgpts[0])
        cv2.line(frame, origin, tuple(imgpts[3]), (0, 0, 255), 2)  # X - red
        cv2.line(frame, origin, tuple(imgpts[2]), (0, 255, 0), 2)  # Y - green
        cv2.line(frame, origin, tuple(imgpts[1]), (255, 0, 0), 2)  # Z - blue

        return frame
