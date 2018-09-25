import numpy as np
from trajectory_pso import Trajectory_PSO, Trajectory_PSO_Utils

start = np.array([0, 0])
end = np.array([1, 1])
num_coordinates = 2
num_waypoints = 8
num_trajectories = 256
num_iters = 300
pso = Trajectory_PSO(num_coordinates, num_waypoints, num_trajectories, Trajectory_PSO_Utils.distances_squared, start, end)
cost, traj = pso.run(num_iters)
pso.visualize(traj)
