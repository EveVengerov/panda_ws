# Generate aruco marker with white border

import cv2
import cv2.aruco as aruco
import numpy as np
import os

# ==== PARAMETERS ====
aruco_dict_type = aruco.DICT_4X4_100  # Choose dictionary
marker_id = 0                         # ID of marker
marker_size = 200                     # Size of marker (inner marker)
border_thickness = 40                 # Thickness of white border in pixels

border_to_marker_ratio = border_thickness/marker_size

# ==== PATH ====
folder_path = "aruco_cube_description/materials/textures/"
os.makedirs(folder_path, exist_ok=True)
save_path = os.path.join(folder_path, f"aruco_{marker_id}_with_border.png")

# ==== CREATE MARKER ====
aruco_dict = aruco.getPredefinedDictionary(aruco_dict_type)
marker_img = aruco.generateImageMarker(aruco_dict, marker_id, marker_size)

# ==== ADD WHITE BORDER ====
bordered_img = cv2.copyMakeBorder(
    marker_img,
    top=border_thickness,
    bottom=border_thickness,
    left=border_thickness,
    right=border_thickness,
    borderType=cv2.BORDER_CONSTANT,
    value=255  # White color
)

# ==== SAVE IMAGE ====
cv2.imwrite(save_path, bordered_img)
print(f"Aruco marker with white border saved to {save_path}")
print(f"Border to marker ratio: {border_to_marker_ratio}")
