from pso_lqr_rrt_star import PSO_LQR_RRT_star
import numpy as np

# LQR_RRT_Star
start = np.array([0, 0])
end = np.array([np.pi, 0])
state_bounds = np.array([[-2 * np.pi, -2.5 * np.pi], [2 * np.pi, 2.5 * np.pi]])
control_bounds = np.array([[-3], [3]])
b = .1
g = 9.81
def A_fn(x, u):
    return np.array([[0, 1],[-g * np.cos(x[0]), -b]])
def B_fn(x, u):
    return np.array([[0], [1]])
obstacles = np.array([[0, 0, 0]])
Q = np.diag([1, 1])
R = np.diag([1])
dt = 0.05
def update_fn(x, u):
  u = np.clip(u, *control_bounds)
  return np.array([x[0] + dt * x[1], x[1] + dt * u[0]])
def traj_cost_fn(traj):
  angles = traj[:, 0]
  deltas = angles[1:] - angles[:-1]
  controls = traj[:, 1]
  return np.sum(deltas ** 2) + np.sum(np.absolute(controls))

num_waypoints = 10
num_trajectories = 256
num_pso_iterations = 300
num_rrt_iterations = 3000


alg = PSO_LQR_RRT_star(start, end, state_bounds, obstacles, A_fn, B_fn, Q, R, update_fn, num_waypoints, num_trajectories, traj_cost_fn)
alg.run(num_pso_iterations, num_rrt_iterations)
alg.visualize()
