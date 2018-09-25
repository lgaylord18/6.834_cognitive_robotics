from lqr_rrt_star import LQR_RRT_star
from dubins import *

alg = LQR_RRT_star(start, end, state_bounds, obstacles, A_fn, B_fn, Q, R, update_fn, obstacle_coords)
alg.run(20000)
alg.visualize(end, .1, [0, 1])
