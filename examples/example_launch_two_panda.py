# EXAMPLES SCRIPT TO TEST TWO PANDA ROBOTS IN PYBULLET

import pybullet as p
import pybullet_data
import numpy as np
import time

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# Load two tables at different y-positions
# Table height is 0.62m, so place robots at z=0.62

tableA = p.loadURDF("table/table.urdf", basePosition=[0.5, -0.6, 0])
tableB = p.loadURDF("table/table.urdf", basePosition=[0.5,  0.6, 0])

# Load two Panda robots with shifted base positions (on top of tables)
robotA = p.loadURDF("franka_panda/panda.urdf", basePosition=[0.5, -0.6, 0.62], useFixedBase=True)
robotB = p.loadURDF("franka_panda/panda.urdf", basePosition=[0.5,  0.6, 0.62], useFixedBase=True)


try :
    while True:
        p.stepSimulation()
        time.sleep(1. / 240.)
except KeyboardInterrupt:
    p.disconnect()