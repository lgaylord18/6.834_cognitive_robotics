from pso_lqr_rrt_star import PSO_LQR_RRT_star
from dubins import *

alg = PSO_LQR_RRT_star(start, end, state_bounds, obstacles, A_fn, B_fn, Q, R, update_fn, num_waypoints, num_trajectories, traj_cost_fn)
alg.run(100, 800)
alg.visualize([0, 1])
