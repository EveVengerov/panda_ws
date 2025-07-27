"""3R Planar Robot Kinematic Redundancy Example"""

"""This example demonstrates the kinematic redundancy of a 3R planar robot arm.
1. specify link lengths and joint limits 
2. compute forward kinematics
3. compute jacobian and inverse jacobian 
4. compte inverse kinematics for a target end-effector position
5. find the null space configurations
6. animate the robot arm and joint angles with different null space configurations
7. visualize the joint angles and end-effector trajectory and joint limits in one graph 




"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Robot Parameters ---
L1, L2, L3 = 1.0, 1.0, 1.0
link_lengths = [L1, L2, L3]
n_joints = 3
target_pos = np.array([1.5, 1.5])  # Static target

# --- Forward Kinematics ---
def forward_kinematics(theta):
    x, y = 0.0, 0.0
    joint_positions = [(x, y)]
    total_angle = 0.0
    for i in range(n_joints):
        total_angle += theta[i]
        x += link_lengths[i] * np.cos(total_angle)
        y += link_lengths[i] * np.sin(total_angle)
        joint_positions.append((x, y))
    return np.array(joint_positions)


# --- Jacobian Calculation ---
def compute_jacobian(q, l=[1, 1, 1]):
    q1, q2, q3 = q
    J = np.zeros((2, 3))
    J[0, 0] = -l[0]*np.sin(q1) - l[1]*np.sin(q1+q2) - l[2]*np.sin(q1+q2+q3)
    J[0, 1] = -l[1]*np.sin(q1+q2) - l[2]*np.sin(q1+q2+q3)
    J[0, 2] = -l[2]*np.sin(q1+q2+q3)
    J[1, 0] =  l[0]*np.cos(q1) + l[1]*np.cos(q1+q2) + l[2]*np.cos(q1+q2+q3)
    J[1, 1] =  l[1]*np.cos(q1+q2) + l[2]*np.cos(q1+q2+q3)
    J[1, 2] =  l[2]*np.cos(q1+q2+q3)
    return J

# --- Inverse Kinematics ---
def inverse_kinematics(target):
    theta = np.zeros(n_joints)
    for _ in range(100):  # Iterative method
        J = compute_jacobian(theta)
        ee_pos = forward_kinematics(theta)[-1]
        error = target - ee_pos
        if np.linalg.norm(error) < 1e-4:
            break
        dtheta = np.linalg.pinv(J) @ error  # Pseudo-inverse for redundancy
        theta += dtheta
    return theta

# --- Numerical IK (Single Point) ---
def numerical_IK(theta_init, target, lr=0.1, max_iter=100, tol=1e-4):
    theta_list = [theta_init.copy()]
    theta = theta_init.copy()
    for _ in range(max_iter):
        joints = forward_kinematics(theta)
        ee_pos = joints[-1]
        error = target - ee_pos
        if np.linalg.norm(error) < tol:
            break

        # Numerical Jacobian
        J = np.zeros((2, n_joints))
        delta = 1e-5
        for i in range(n_joints):
            theta_perturbed = theta.copy()
            theta_perturbed[i] += delta
            perturbed_ee = forward_kinematics(theta_perturbed)[-1]
            J[:, i] = (perturbed_ee - ee_pos) / delta

        # Gradient descent step
        dtheta = J.T @ error
        theta += lr * dtheta
        theta_list.append(theta.copy())
    return theta_list

def pseudo_inverse(J):
    return np.linalg.pinv(J)



def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

# Example for vector of joint angles
def wrap_angles(q):
    return np.array([wrap_to_pi(a) for a in q])

def ik_with_null_space(q, target_pos, q_rest, alpha=0.1, max_iter=100, tol=1e-2):
        theta = np.zeros(n_joints)
        identity = np.eye(n_joints)
        
        for i in range(max_iter):  # Iterative method
            J = compute_jacobian(theta)
            ee_pos = forward_kinematics(theta)[-1]
            error = target_pos - ee_pos
            if np.linalg.norm(error) < 1e-4:
                break
            
            null_proj = identity - np.linalg.pinv(J) @ J
            gradient = -alpha * (q - q_rest)
            secondary = null_proj @ gradient
            
            dtheta = np.linalg.pinv(J) @ error + secondary # Pseudo-inverse for redundancy
            theta += dtheta
            theta = wrap_angles(theta)  # Ensure angles are within [-pi, pi]

            # if i % 10 == 0:
            #     print(f"Iteration {i}, Error: {np.linalg.norm(error)}, Joint Angles: {theta}")

        return theta

# --- Compute IK Steps ---
theta_init = np.zeros(n_joints)

q_rest = np.array([0, 0, 0])  # Rest position for null space
nullspace_goals = [
    q_rest,
    np.array([0.0, -np.pi/4, 0.0]),
    np.array([np.pi/2, 0.0, -np.pi/2]),
    np.array([-np.pi/2, np.pi/4, np.pi/2])
]

theta_steps = []

# Compute IK for the target position with different null space goals
for null_space_goal in nullspace_goals:
    print(f"Computing IK with null space goal: {null_space_goal}")
    theta_target = ik_with_null_space(theta_init, target_pos, null_space_goal)
    print(f"Target angles: {theta_target}")
    theta = numerical_IK(theta_target, target_pos)
    np.append(theta_steps, theta)  # Store the last angle configuration
    
    theta_steps.append(theta)



# --- Plot Setup ---
fig, ax = plt.subplots()
ax.set_xlim(-3.5, 3.5)
ax.set_ylim(-3.5, 3.5)
ax.set_aspect('equal')
ax.grid(True)
line, = ax.plot([], [], 'o-', lw=4)
joint_dot, = ax.plot([], [], 'bo', markersize=8)
ax.set_title("3R Planar Robot - Numerical IK to Single Target")

ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")

ax.plot(target_pos[0], target_pos[1], 'rx', markersize=12, lin, label='Target Position')
plot_colors = ['b', 'g', 'c', 'm', 'y', 'k']

for i, theta, null_space_goal in zip(range(len(theta_steps)), theta_steps, nullspace_goals):
    print(f"Null Space Solution {i}: {theta[0]}") # list  of np arrays
    points = forward_kinematics(theta[0])
    theta_angle = np.degrees(theta[0])
    ax.plot(points[:, 0], points[:, 1], label=f'Null Space Solution: {theta_angle}', color=plot_colors[i])
    ax.plot(points[:, 0], points[:, 1], plot_colors[i]+'o', markersize=4)
    
    points = forward_kinematics(null_space_goal)
    ax.plot(points[:, 0], points[:, 1], "--" + plot_colors[i], alpha=0.5, label=f'Null Space Goal: {null_space_goal}')
    ax.plot(points[:, 0], points[:, 1], plot_colors[i]+'o', markersize=4)


        
# for i, null_space_goal in zip(range(len(nullspace_goals)), nullspace_goals):
#     print(f"Null Space Goal: {null_space_goal[0]}")
#     points = forward_kinematics(null_space_goal)
#     ax.plot(points[:, 0], points[:, 1], "--" + plot_colors[i], alpha=0.5, label=f'Null Space Goal: {null_space_goal}')
#     ax.plot(points[:, 0], points[:, 1], plot_colors[i]+'o', markersize=4)

ax.legend()

# # --- Animation Update Function ---
# def update(frame):
#     theta = theta_steps[frame]
#     points = forward_kinematics(theta)
#     line.set_data(points[:, 0], points[:, 1])
#     target_dot.set_data(target_pos[0], target_pos[1])
#     return line, target_dot

# ani = FuncAnimation(fig, update, frames=len(theta_steps), interval=150, blit=True)
plt.show()
