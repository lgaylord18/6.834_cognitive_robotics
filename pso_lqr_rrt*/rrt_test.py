import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from rrt import RRT

start = np.array([0, 0])
end = np.array([1, 1])
bounds = np.array([[-2, -2], [2, 2]])
obstacles = np.array([[.5, .5, .2]])
alg = RRT(start, end, bounds, obstacles)
alg.run(3000)
alg.visualize(end, .1)
