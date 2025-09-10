import pybullet as p
import pybullet_data
import time
import numpy as np
import matplotlib.pyplot as plt

# Connect to PyBullet
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load Panda robot
robot_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

# Get joint info
num_joints = p.getNumJoints(robot_id)
# The Panda arm has 7 revolute joints.
joint_indices = [i for i in range(num_joints) if p.getJointInfo(robot_id, i)[2] == p.JOINT_REVOLUTE]
print(f"Revolute joint indices: {joint_indices}")


# step position declaration
step_positions = [0.0] * len(joint_indices)

# Get and print joint limits
print("\n--- Joint Limits ---")
for idx in joint_indices:
    joint_info = p.getJointInfo(robot_id, idx)
    joint_name = joint_info[1].decode('utf-8')
    lower_limit = joint_info[8]
    upper_limit = joint_info[9]
    print(f"Joint {idx} ({joint_name}): Lower={lower_limit:.2f}, Upper={upper_limit:.2f}")
    if lower_limit + upper_limit == 0:
        print(f"  Warning: Joint {idx} has symmetric limits around zero. Setting step position to 0.1 rad.")
        step_positions[idx] = 0.1
    else:
        step_positions[idx] = (lower_limit + upper_limit) / 2
print("--------------------\n")

# Reset all joints to zero position
for idx in joint_indices:
    p.resetJointState(robot_id, idx, 0)

# Step response parameters
# step_position = 50*(np.pi / 180)  # radians
duration = 1.0       # seconds
dt = 0.01            # timestep
steps = int(duration / dt)

# Apply step input and record response
joint_positions = {idx: [] for idx in joint_indices}

# Apply step input at t=0
# target_positions = [step_position] * len(joint_indices)
p.setJointMotorControlArray(
    bodyUniqueId=robot_id,
    jointIndices=joint_indices,
    controlMode=p.POSITION_CONTROL,
    targetPositions=step_positions
)

t = np.linspace(0, duration, steps)
for step in range(steps):
    p.setJointMotorControlArray(
    bodyUniqueId=robot_id,
    jointIndices=joint_indices,
    controlMode=p.POSITION_CONTROL,
    targetPositions=step_positions
)
    p.stepSimulation()
    
    joint_states = p.getJointStates(robot_id, joint_indices)
    for i, idx in enumerate(joint_indices):
        pos = joint_states[i][0]
        joint_positions[idx].append(pos)
        
    p.setTimeStep(dt)
    time.sleep(dt)

# # Print results
# for idx in joint_indices:
#     print(f"Joint {idx} step response (first 10 samples): {joint_positions[idx][:10]}")

# --- Analysis Functions ---
def compute_step_info(t, y, final_value, settling_tolerance=0.02):
    """
    Compute rise time and settling time for a step response.
    """
    # Rise time: time to go from 10% to 90% of final value
    try:
        if final_value > 0:
            t_10 = t[np.where(y >= 0.1 * final_value)[0][0]]
            t_90 = t[np.where(y >= 0.9 * final_value)[0][0]]
            rise_time = t_90 - t_10
        else:
            t_10 = t[np.where(y <= 0.1 * final_value)[0][0]]
            t_90 = t[np.where(y <= 0.9 * final_value)[0][0]]
            rise_time = t_90 - t_10
    
    except IndexError:
        rise_time = None

    # Settling time: time after which the response is within a tolerance band
    settling_time = None
    tolerance_band = final_value * settling_tolerance
    # Find the last time the response is outside the tolerance band
    outside_tolerance = np.where(np.abs(y - final_value) > tolerance_band)[0]
    if len(outside_tolerance) > 0:
        last_outside_idx = outside_tolerance[-1]
        if last_outside_idx + 1 < len(t):
            settling_time = t[last_outside_idx + 1]
    else: # Already within tolerance
        settling_time = t[0]
        
    return rise_time, settling_time

# --- Plotting ---
fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
axes = axes.ravel() # Flatten the 3x2 grid to a 1D array
# t = np.linspace(0, duration, steps)

rise_time_max = -np.inf
for i, idx in enumerate(joint_indices[:6]): # Plot first 6 joints
    ax = axes[i]
    response = np.array(joint_positions[idx])
    
    # Plot response
    ax.plot(t, response, label=f'Joint {idx} Response')
    
    # Plot target line
    ax.axhline(y=step_positions[idx], color='r', linestyle='--', label='Target')

    # Compute and display metrics
    rise_time, settling_time = compute_step_info(t, response, step_positions[idx])
    
    if rise_time is not None and rise_time > rise_time_max: 
        rise_time_max = rise_time

    info_text = []
    if rise_time is not None:
        info_text.append(f'Rise Time: {rise_time:.2f}s')
    if settling_time is not None:
        info_text.append(f'Settling Time: {settling_time:.2f}s')
        # Shade the settling region
        ax.axvspan(settling_time, t[-1], color='green', alpha=0.2, label=f'Settled (±2%)')

    # Add text box with info
    ax.text(0.5, 0.1, '\n'.join(info_text), transform=ax.transAxes, 
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))

    ax.set_title(f'Step Response of Joint {idx}')
    ax.set_ylabel('Position (rad)')
    ax.grid(True)
    ax.legend()

# Add a shared x-label
fig.text(0.5, 0.04, 'Time (s)', ha='center', va='center')
plt.tight_layout(rect=[0, 0.05, 1, 1]) # Adjust layout to make room for x-label
plt.show()

p.disconnect()

print("-"*10, "Summary", "-"*10)
print("Simulation speed: ",  1/dt, "Hz")
print("Maximum Rise Time (s) :", rise_time_max)
print("Approximate 3db cutoff frequency (Hz):", 0.35/rise_time_max if rise_time_max else None)
print("Step response analysis complete.")