import pybullet as p
import pybullet_data


def setup_pybullet():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")

def load_tables():
    table_position_a = [0.5, -0.6, 0]
    table_position_b = [0.5,  0.6, 0]
    tableA = p.loadURDF("table/table.urdf", basePosition=table_position_a)
    tableB = p.loadURDF("table/table.urdf", basePosition=table_position_b)
    table_offset = 0.62
    return table_position_a, table_position_b, tableA, tableB, table_offset

def load_robots(table_position_a, table_position_b, table_offset):
    robotA = p.loadURDF("franka_panda/panda.urdf", basePosition=[table_position_a[0], table_position_a[1], table_offset], useFixedBase=True)
    robotB = p.loadURDF("franka_panda/panda.urdf", basePosition=[table_position_b[0], table_position_b[1], table_offset], useFixedBase=True)
    return robotA, robotB

def load_cube(cube_start, scale_to_meters):
    cube_id = p.loadURDF(
        "aruco_cube_description/urdf/aruco.urdf",
        cube_start,
        globalScaling=scale_to_meters
    )
    texture_id = p.loadTexture("aruco_cube_description/materials/textures/aruco_0_with_border.png")
    p.changeVisualShape(cube_id, -1, textureUniqueId=texture_id)
    return cube_id
