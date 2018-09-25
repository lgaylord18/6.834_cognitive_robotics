import numpy as np
from rrt_star import Node
from lqr_rrt_star import LQR_RRT_star
from trajectory_pso import Trajectory_PSO, Trajectory_PSO_Utils


class PSO_LQR_RRT_star:
    def __init__(self, s_init, s_goal, bounds, obstacles, A_fn, B_fn, Q, R, update_fn, num_waypoints, num_trajectories, traj_cost_fn, obstacle_coords = None, gamma = 1):
        self.lqr_rrt_star = LQR_RRT_star(s_init, s_goal, bounds, obstacles, A_fn, B_fn, Q, R, update_fn, obstacle_coords, gamma)
        self.pso = Trajectory_PSO(s_init.shape[0], num_waypoints, num_trajectories, traj_cost_fn, s_init, s_goal)
        self.start_state = s_init
        self.goal_state = s_goal

    def run(self, num_pso_iterations, num_rrt_iterations):
        if self.lqr_rrt_star.T.number_of_nodes() > 1:
            self.lqr_rrt_star.T.clear()
            self.lqr_rrt_star.T.add_node(self.lqr_rrt_star.start_node)
        # run PSO
        cost, traj = self.pso.run(num_pso_iterations)
        self.pso_traj = traj
        # add trajectory to RRT
        traj = np.vstack((traj, self.goal_state))
        last_node = self.lqr_rrt_star.start_node
        for state in traj:
            new_node = Node(state)
            self.lqr_rrt_star.extend(last_node, new_node)
            last_node = new_node
        # run RRT
        self.lqr_rrt_star.run(num_rrt_iterations)
        return self.lqr_rrt_star.best_path(self.goal_state, 0)

    def visualize(self, plot_coordinates = None):
        self.pso.visualize(self.pso_traj, plot_coordinates)
        self.lqr_rrt_star.visualize(self.goal_state, 0, plot_coordinates)
