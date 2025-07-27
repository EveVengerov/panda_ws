"""
=============================================================
Script Summary: Panda Lissajous Trajectory Demo
=============================================================
This script demonstrates the Franka Panda robot following a Lissajous trajectory in PyBullet.

Main Features:
- Loads a single Panda robot in a simulated environment.
- Sets the robot to a home pose.
- Generates a Lissajous curve trajectory for the end-effector (in the XZ plane by default).
- Uses inverse kinematics (IK) to compute joint angles for each target point.
- Visualizes the end-effector path with debug lines in the simulation.
- Optionally plots the desired trajectory using matplotlib.

Usage:
    python launch_panda_lissajous.py
=============================================================
"""
import numpy as np
import pybullet as p
import pybullet_data
import time
import signal

from lib.trajectory_generator import CircleTrajectoryGenerator, LissajousTrajectoryGenerator

# Install a SIGINT handler to ensure KeyboardInterrupt works properly
import signal
signal.signal(signal.SIGINT, signal.default_int_handler)

try:
    # — Setup PyBullet —
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

    # — Joint and IK setup —
    n = 7
    ee = 11
    infos = [p.getJointInfo(robot, i) for i in range(n)]
    lower = [info[8] for info in infos]
    upper = [info[9] for info in infos]
    ranges = [upper[i] - lower[i] for i in range(n)]
    rest = [0]*n

    # — Home pose —
    home = [0, -np.pi/4, 0, -np.pi/2, 0, np.pi/3, 0]
    for i, q in enumerate(home):
        p.resetJointState(robot, i, q)
    time.sleep(1)

    # — Trajectory parameters — Circle Trajectory
    # center = np.array([0.5, 0, 0.5])
    # radius = 0.1
    # T = 8.0
    # steps = 400
    # dt = 0.01
    
    # circle_trajectory = CircleTrajectoryGenerator(center, radius, T, steps, plane='xz')
        
        
    # Trajectory parameters - Lissajous Trajectory
    A, B = 0.3, 0.3
    a = 1
    b = 2
    delta = np.pi / 32
    center = np.array([ 0, 0.5, 0.5])
    # get greatest command divisor of a and b 
    
    g = np.gcd(a, b)
    print(g)
    T = 2*np.pi/g
    dt = 1/240   
    steps = int(T/dt) + 1
    
    print(f"Time Period: {T}, Steps: {steps}, dt: {dt}")

    
    lissajous_trajectory = LissajousTrajectoryGenerator(A, B, a, b, delta, center, T, steps, plane='xz')
    lissajous_trajectory.get_plot()
    
    print(f"Total steps for trajectory: {steps}")
    
    prev_pos = np.array(p.getLinkState(robot, ee)[0])

    # # --- Plot desired trajectory in simulation ---
    # print("📈 Plotting desired trajectory in simulation...")
    # trajectory_points = [lissajous_trajectory.get_target_position(i) for i in range(steps)]
    # for i in range(1, len(trajectory_points)):
    #     p.addUserDebugLine(trajectory_points[i-1], trajectory_points[i], [0, 1, 0], 1, lifeTime=0)

    print("🔁 Running trajectory. Press Ctrl+C to stop at any time.")

    p.setTimeStep(1/240)

    k = 0
    while True:
    # for k in range(0, steps):

        # target = circle_trajectory.get_target_position(k)
        target = lissajous_trajectory.get_target_position(k)
        
        ori = p.getQuaternionFromEuler([0, np.pi/2, 0])

        angles = p.calculateInverseKinematics(
            robot, ee, target,
            lowerLimits=lower, upperLimits=upper,
            jointRanges=ranges, restPoses=rest
        )[:n]

        p.setJointMotorControlArray(
            robot, range(n),
            p.POSITION_CONTROL,
            targetPositions=angles,
            targetVelocities=[0]*n,
            forces=[90]*n
        )

        p.stepSimulation()
        time.sleep(dt)

        curr_pos = np.array(p.getLinkState(robot, ee)[0])
        p.addUserDebugLine(prev_pos, curr_pos, [1, 0, 0], 2, lifeTime=0)
        prev_pos = curr_pos
        k = k + 1
        
    print("✅ Trajectory complete.")
    time.sleep(2)

except KeyboardInterrupt:
    print("\n⏹ User interrupted with Ctrl+C — stopping simulation.")

finally:
    p.disconnect()
    print("Simulation disconnected. Cleanup complete.")
