import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from lqr_rrt_star import LQR_RRT_star

start = np.array([0, 0])
end = np.array([np.pi, 0])
state_bounds = np.array([[-2 * np.pi, -2.5 * np.pi], [2 * np.pi, 2.5 * np.pi]])
control_bounds = np.array([[-5], [5]])
b = .1
g = 9.81
def A_fn(x, u):
    return np.array([[0, 1],[-g * np.cos(x[0]), -b]])
def B_fn(x, u):
    return np.array([[0], [1]])
obstacles = np.array([[0, 0, 0]])
Q = np.diag([.5, .05])
R = np.diag([1])
dt = 0.05
def update_fn(x, u):
    u = np.clip(u, *control_bounds)
    return np.array([x[0] + dt * x[1], x[1] + dt * u[0]])
alg = LQR_RRT_star(start, end, state_bounds, obstacles, A_fn, B_fn, Q, R, update_fn)
alg.run(5000)
alg.visualize(end, .2)
