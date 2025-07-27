import numpy as np 
from scipy.spatial.transform import Rotation as R

class CircleTrajectoryGenerator:
    def __init__(self, center, radius, T, steps, plane='xy'):
        self.center = np.array(center)
        self.radius = radius
        self.T = T
        self.steps = steps
        self.dt = T / steps
        self.plane = plane

    def get_target_position(self, k):
        theta = 2 * np.pi * k / self.steps
        
        if self.plane == 'xy':
            return self.center + self.radius * np.array([np.cos(theta), np.sin(theta), 0])
        elif self.plane == 'xz':
            return self.center + self.radius * np.array([np.cos(theta), 0, np.sin(theta)])
        elif self.plane == 'yz':
            return self.center + self.radius * np.array([0, np.cos(theta), np.sin(theta)])
        else:
            raise ValueError("Invalid circle plane. Choose from 'xy', 'xz', or 'yz'.")
        
class LissajousTrajectoryGenerator:
    def __init__(self, A, B, a, b, delta, center, T, steps, plane='xy'):
        
        self.A = A
        self.B = B
        self.a = a
        self.b = b
        self.delta = delta
        self.center = np.array(center)
        self.T = T
        self.steps = steps
        self.dt = T / steps
        self.plane = plane

    def get_target_position(self, k):
        t = k * self.dt
        x = self.A * np.sin(self.a*t) 
        y = self.B * np.sin(self.b*t + self.delta)
        
        if self.plane == 'xy':
            return self.center + np.array([x, y, 0])
        elif self.plane == 'xz':
            return self.center + np.array([x, 0, y])
        elif self.plane == 'yz':
            return self.center + np.array([0, x, y])
        else:
            raise ValueError("Invalid Lissajous plane. Choose from 'xy', 'xz', or 'yz'.")

    def get_plot(self):
        import matplotlib.pyplot as plt
        t = np.linspace(0, self.T, self.steps)
        x = self.center[0] + self.A * np.sin(self.a*t) 
        y = self.center[1] + self.B * np.sin(self.b*t + self.delta)
        
        if self.plane == 'xy':
            plt.plot(x, y)
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
        elif self.plane == 'xz':
            plt.plot(x, y)
            plt.xlabel('X-axis')
            plt.ylabel('Z-axis')
        elif self.plane == 'yz':
            plt.plot(y, x)
            plt.xlabel('Y-axis')
            plt.ylabel('Z-axis')
        
        plt.title(f'Lissajous Curve in {self.plane.upper()} Plane')
        plt.grid()
        plt.axis('equal')
        plt.show()
        
        
class MinimumJerkTrajectoryGenerator:
    def __init__(self, start, end, duration, dt):
        self.start = np.array(start)
        self.end = np.array(end)
        self.T = duration
        self.steps = int(duration / dt)
        self.dt = dt

    def get_target_position(self, k):
        t = k * self.dt
        
        # return positiiona nd velocity at time t
        if k >= self.steps:
            return self.end, np.zeros(np.shape(self.end))
        
        position = self.start + (self.end - self.start) * (10*(t/self.T)**3 - 15*(t/self.T)**4 + 6*(t/self.T)**5)
        velocity = (self.end - self.start) * (30*(t/self.T)**2 - 60*(t/self.T)**3 + 30*(t/self.T)**4) / self.T  
        return position, velocity
        
    
class TrapezoidalTrajectoryGenerator:
    def __init__(self, start, end, T, steps):
        self.start = np.array(start)
        self.end = np.array(end)
        self.T = T
        self.steps = steps
        self.dt = T / steps

    def get_target_position(self, k):
        t = k * self.dt
        if t < self.T / 2:
            return self.start + (self.end - self.start) * (t / (self.T / 2))
        else:
            return self.end - (self.end - self.start) * ((t - self.T / 2) / (self.T / 2))

    def get_target_position_list(self):
        target_position_list = np.zeros(self.steps)
        for step in self.steps:
            t = step * self.dt
        if t < self.T / 2:
            target_position_list[step] =  self.start + (self.end - self.start) * (t / (self.T / 2))
        else:
            target_position_list[step] =  self.end - (self.end - self.start) * ((t - self.T / 2) / (self.T / 2))

        return target_position_list


class SlerpOrientationTrajectoryGenerator:
    """
    Generates a smooth orientation trajectory between two quaternions using SLERP (Spherical Linear Interpolation).

    Args:
        start_quat: array-like, shape (4,) - Starting quaternion (x, y, z, w) or (w, x, y, z)
        end_quat: array-like, shape (4,) - Ending quaternion
        duration: float - Total duration of the trajectory
        steps: int - Number of interpolation steps

    Usage:
        traj = SlerpOrientationTrajectoryGenerator(start_quat, end_quat, duration, steps)
        for k in range(steps):
            quat = traj.get_target_orientation(k)
    """
    def __init__(self, start_quat, end_quat, duration, steps):
        self.start_quat = np.array(start_quat)
        self.end_quat = np.array(end_quat)
        self.duration = duration
        self.steps = steps
        self.dt = duration / steps
        self.rot_start = R.from_quat(self.start_quat)
        self.rot_end = R.from_quat(self.end_quat)

    def get_target_orientation(self, k):
        """
        Returns the interpolated quaternion at step k.
        """
        if k >= self.steps:
            return self.end_quat
        t = k / (self.steps - 1)
        # Use scipy's Slerp for interpolation
        from scipy.spatial.transform import Slerp
        slerp = Slerp([0, 1], R.from_quat([self.start_quat, self.end_quat]))
        
        rot_interp = slerp([t])[0]
        return rot_interp.as_quat()


def generate_minimum_jerk_and_slerp_trajectory(waypoints, orientations, duration, dt):
    """
    Given a list of waypoints and orientations (quaternions), generate interpolated positions and orientations
    using minimum jerk for position and SLERP for orientation between each consecutive pair.
    Returns:
        positions: list of np.array, interpolated positions
        orientations: list of np.array, interpolated quaternions
    """
    positions = []
    orients = []
    n_segments = len(waypoints) - 1
    steps_per_segment = int(duration / dt // n_segments)
    for i in range(n_segments):
        pos_traj = MinimumJerkTrajectoryGenerator(
            start=waypoints[i],
            end=waypoints[i+1],
            duration=duration/n_segments,
            dt=dt
        )
        slerp_traj = SlerpOrientationTrajectoryGenerator(
            start_quat=orientations[i],
            end_quat=orientations[i+1],
            duration=duration/n_segments,
            steps=steps_per_segment
        )
        for k in range(steps_per_segment):
            pos, _ = pos_traj.get_target_position(k)
            quat = slerp_traj.get_target_orientation(k)
            positions.append(pos)
            orients.append(quat)
    # Ensure last waypoint is included
    positions.append(np.array(waypoints[-1]))
    orients.append(np.array(orientations[-1]))
    return positions, orients