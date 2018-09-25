import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Node(object):
    num_nodes_created = 0

    def __init__(self, state):
        Node.num_nodes_created += 1
        self.id = Node.num_nodes_created
        self.state = state

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return self.id


# state is a d-dimensional numpy array
# bounds is a 2 x d numpy array.
#   First row is min vals for each coordinate
#   Second row is max vals for each coordinate
# obstacles is n x (d + 1)
#   each row is one obstacle
#   first d columns is center point
#   last column is radius
class RRT(object):
    def __init__(self, start_state, goal_state, bounds, obstacles, obstacle_coords=None):
        self.start_state = start_state
        self.goal_state = goal_state
        self.bounds = bounds
        self.obstacle_coords = obstacle_coords
        if obstacle_coords is None:
            self.obstacle_coords = list(range(obstacles.shape[1] - 1))
        self.obstacles = obstacles
        self.T = nx.DiGraph()
        self.start_node = Node(start_state)
        self.T.add_node(self.start_node)
        self.bound_size = bounds[1] - bounds[0]

    # adds new_node to the tree and adds an edge from tree_node to new_node
    def extend(self, tree_node, new_node):
        self.T.add_node(new_node)
        self.T.add_edge(tree_node, new_node)

    # returns a random possible state
    def random(self):
        state = None
        while state is None or self.in_obstacle(state):
            if np.random.random() > .05: # TODO: configurable
                state = self.bounds[0] + np.random.random_sample(self.start_state.shape) * self.bound_size
            else:
                state = self.goal_state
        return state

    # returns true if the inputted state is in an obstacle
    def in_obstacle(self, state):
        if np.any(state < self.bounds[0]) or np.any(state > self.bounds[1]):
            return True
        obstacle_centers = self.obstacles[:,:-1]
        radiuses = self.obstacles[:,-1]
        displacements = obstacle_centers -  state[self.obstacle_coords]
        distances = np.linalg.norm(displacements, axis=1)
        return np.any(distances < radiuses)

    # returns true if the path from from_state to to_state goes through an
    #     obstacle
    # detection done as described below
    # http://mathworld.wolfram.com/Circle-LineIntersection.html
    # not 100% sure this works in 3d but that doesn't matter for our purposes
    def through_obstacle(self, from_state, to_state):
        if self.in_obstacle(from_state) or self.in_obstacle(to_state):
            return True
        a = from_state[self.obstacle_coords]
        b = to_state[self.obstacle_coords]
        c = self.obstacles[:,:-1]
        r2 = self.obstacles[:,-1]**2
        d2 = np.sum(np.abs(b - a)**2) # norm squared
        x = np.cross(-c + a, -c + b, axis=1)
        if (len(x.shape) == 1):
            x = x.reshape((-1, 1))
        D = np.linalg.norm(x, axis=1)
        if np.any(d2 * r2 - D**2 > 0):
            return True
        return False

    # returns the euclidean distance between two nodes
    def distance(self, state_u, state_v):
        return np.linalg.norm(state_u - state_v)

    # returns the node in the tree which is closest to to_state
    def nearest(self, to_state):
        min_distance = np.infty
        min_node = None
        for tree_node in self.T.nodes:
            distance = self.distance(tree_node.state, to_state)
            if distance < min_distance:
                min_distance = distance
                min_node = tree_node
        return min_node

    # returns a state which is close to from_state and in the direction of
    #     to_state from from_state
    def steer(self, from_state, to_state):
        max_dist_for_steer = .05 # TODO: configurable
        displacement = to_state - from_state
        direction = displacement / np.linalg.norm(displacement)
        return from_state + max_dist_for_steer * direction

    # run RRT for num_iters
    def run(self, num_iters):
        for i in range(num_iters):
            x_rand = self.random()
            n_nearest = self.nearest(x_rand)
            x_new = self.steer(n_nearest.state, x_rand)
            if not self.through_obstacle(n_nearest.state, x_new):
                n_new = Node(x_new)
                self.extend(n_nearest, n_new)

    # finds the shortest path form the start node to a node within radius
    #   of the goal state (using euclidean distance)
    def best_path(self, goal_state, radius):
        best_path = []
        best_path_len = np.infty
        paths = nx.shortest_path(self.T, source=self.start_node)
        for target in paths.keys():
            if np.linalg.norm(target.state - goal_state) <= radius:
                path = paths[target]
                path_len = len(path)
                if path_len < best_path_len:
                    best_path_len = path_len
                    best_path = path
        if not best_path:
            print("No path found.")
        return [n.state for n in best_path]

    def visualize(self, goal_state, tolerance, plot_coordinates = None):
        if plot_coordinates is None:
            plot_coordinates = list(range(self.start_state.shape[0]))
        num_coordinates = len(plot_coordinates)
        if num_coordinates != 2 and num_coordinates  != 3:
            print("Cannot visualize in {} dimensions.".format(num_coordinates))
        fig = plt.figure("Results Tree")
        ax = fig.gca(projection=('3d' if num_coordinates == 3 else None))
        for u, v in self.T.edges:
            u_coords = u.state[plot_coordinates].tolist()
            v_coords = v.state[plot_coordinates].tolist()
            coords = np.vstack((u_coords, v_coords)).T
            ax.plot(*coords, color=str(np.random.random()))
        best_path = self.best_path(goal_state, tolerance)
        if best_path:
          for i in range(len(best_path) - 1):
              x_1 = best_path[i][plot_coordinates].tolist()
              x_2 = best_path[i + 1][plot_coordinates].tolist()
              states = np.vstack((x_1, x_2)).T
              ax.plot(*states, color='g')
        start_coords = self.start_state[plot_coordinates].tolist()
        goal_coords = goal_state[plot_coordinates].tolist()
        ax.scatter(*start_coords, color='b')
        ax.scatter(*goal_coords, color='b')
        if num_coordinates == 2:
            goal_circle = plt.Circle(goal_coords, tolerance, color='b')
            goal_circle.set_facecolor('none')
            ax.add_patch(goal_circle)
            if self.obstacle_coords == plot_coordinates:
                for obstacle in self.obstacles:
                    radius = obstacle[-1]
                    center = obstacle[:-1]
                    ax.add_patch(plt.Circle(center, radius, color='r'))
        elif num_coordinates == 3:
            pass # TODO: spheres
        plt.title('RRT Trajectory')
        plt.show()
