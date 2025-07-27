"""
Lissajous Trajectory Parameter Estimation using Nonlinear Least Squares
=====================================================================
This script estimates the parameters of a Lissajous trajectory (A, B, a, b, delta, center)
given noisy position data using nonlinear least squares optimization.

The Lissajous trajectory is defined as:
    x(t) = center_x + A * sin(a * t + delta)
    z(t) = center_z + B * sin(b * t)

Usage:
    python estimate_lissajous_params.py

Dependencies:
    numpy, scipy, matplotlib
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# --- Generate synthetic noisy data for demonstration ---
np.random.seed(42)
A_true, B_true = 0.3, 0.2
center_true = np.array([0.5, 0.0, 0.5])
a_true, b_true = 1, 2
delta_true = np.pi / 8
T = 2 * np.pi
N = 200

t = np.linspace(0, T, N)
x = center_true[0] + A_true * np.sin(a_true * t + delta_true)
z = center_true[2] + B_true * np.sin(b_true * t)

# Add noise
noise_level = 0.01
x_noisy = x + np.random.normal(0, noise_level, size=N)
z_noisy = z + np.random.normal(0, noise_level, size=N)

# --- Parameter estimation function ---
def lissajous_model(t, params):
    A, B, a, b, delta, cx, cz = params
    return np.stack([
        cx + A * np.sin(a * t + delta),
        cz + B * np.sin(b * t)
    ], axis=1)

def residuals(params, t, data):
    pred = lissajous_model(t, params)
    return (pred - data).ravel()

# Initial guess
params0 = [0.2, 0.1, 1, 2, 0, 0.4, 0.4]
data = np.stack([x_noisy, z_noisy], axis=1)

result = least_squares(residuals, params0, args=(t, data))
A_est, B_est, a_est, b_est, delta_est, cx_est, cz_est = result.x

print("Estimated parameters:")
print(f"A = {A_est:.4f}, B = {B_est:.4f}, a = {a_est:.4f}, b = {b_est:.4f}, delta = {delta_est:.4f}, center = [{cx_est:.4f}, {cz_est:.4f}]")

# --- Plot results ---
plt.figure(figsize=(8, 5))
plt.scatter(x_noisy, z_noisy, s=10, label='Noisy Data', alpha=0.6)
t_fit = np.linspace(0, T, 500)
x_fit, z_fit = lissajous_model(t_fit, result.x).T
plt.plot(x_fit, z_fit, 'r-', label='Fitted Lissajous')
plt.xlabel('x')
plt.ylabel('z')
plt.title('Lissajous Trajectory Parameter Estimation')
plt.legend()
plt.axis('equal')
plt.tight_layout()
plt.show()
