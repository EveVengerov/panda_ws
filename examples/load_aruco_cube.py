import pybullet as p
import pybullet_data
import time

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.loadURDF("plane.urdf")

# The mesh in my URDF is scaled 0.05, so to use globalScaling ...
scale = 0.05
cube_size_meters = 0.5
scale_to_meters = cube_size_meters/scale

cube_id = p.loadURDF(
    "aruco_cube_description/urdf/aruco.urdf",
    globalScaling=scale_to_meters
)

texture_id = p.loadTexture("aruco_cube_description/materials/textures/aruco_0_with_border.png")
p.changeVisualShape(cube_id, -1, textureUniqueId=texture_id)

try:
    while True:
        p.stepSimulation()
        time.sleep(1. / 240.)
except KeyboardInterrupt:
    p.disconnect()