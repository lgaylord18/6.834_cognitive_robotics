import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from rrt_star import RRT_star

start = np.array([0, 0, 0])
end = np.array([1, 1, 1])
bounds = np.array([[-1, -1, -1], [2, 2, 2]])
obstacles = np.array([[.5, .5, .5, .5]])
alg = RRT_star(start, end, bounds, obstacles)
alg.run(5000)
alg.visualize(end, .1)
