import pyswarms as ps
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# if start state and goal state are given it adds these to every trajectory
class Trajectory_PSO:
    def __init__(self, num_coordinates, num_waypoints, num_trajectories, traj_cost_fn, start_state=None, goal_state=None):
        self.num_coordinates = num_coordinates
        self.num_waypoints = num_waypoints
        self.num_trajectories = num_trajectories
        self.traj_cost_fn = traj_cost_fn
        self.start_state = start_state
        self.goal_state = goal_state
        dims = (num_waypoints * num_coordinates)
        options = {'c1': 1.496, 'c2': 1.496, 'w':0.7298} # TODO: configurable
        self.pso = ps.single.GlobalBestPSO(n_particles=num_trajectories, dimensions=dims, options=options)

    def add_start_and_goal(self, trajectory):
        if not (self.start_state is None) and not (self.goal_state is None):
            return np.vstack((self.start_state, trajectory, self.goal_state))
        if not (self.start_state is None):
            return np.vstack((self.start_state, trajectory))
        if not (self.goal_state is None):
            return np.vstack((trajectory, self.goal_state))
        return trajectory
    
    def particle_cost_fn(self, particle):
        trajectory = particle.reshape((self.num_waypoints, self.num_coordinates))
        trajectory = self.add_start_and_goal(trajectory)
        return self.traj_cost_fn(trajectory)

    def objective(self, particles):
        return np.apply_along_axis(self.particle_cost_fn, 1, particles)

    def run(self, num_iters):
        cost, particle = self.pso.optimize(self.objective, iters=num_iters)
        trajectory = particle.reshape((self.num_waypoints, self.num_coordinates))
        return cost, self.add_start_and_goal(trajectory)

    def visualize(self, trajectory, plot_coordinates = None):
        if plot_coordinates is None:
            plot_coordinates = list(range(self.num_coordinates))
        num_coordinates = len(plot_coordinates)
        if num_coordinates == 2 or num_coordinates == 3:
            trajectory = self.add_start_and_goal(trajectory)
            fig = plt.figure()
            ax = fig.gca(projection=('3d' if num_coordinates == 3 else None))
            points = map(lambda i: trajectory[:,i], plot_coordinates)
            ax.plot(*points, label='parametric curve', color='g')
            if not (self.start_state is None):
                ax.scatter(*self.start_state[plot_coordinates], color='b')
            if not (self.goal_state is None):
                ax.scatter(*self.goal_state[plot_coordinates], color='b')
            plt.title('PSO Trajectory')
            plt.show()
        else:
            print("Cannot visualize in {} dimensions.".format(trajectory.shape[1]))

class Trajectory_PSO_Utils:
    def length(trajectory):
        distances = np.linalg.norm(trajectory[1:] - trajectory[:-1], axis=1)
        return np.sum(distances)

    def distances_squared(trajectory):
        distances = np.linalg.norm(trajectory[1:] - trajectory[:-1], axis=1)
        return np.sum(distances**2)
