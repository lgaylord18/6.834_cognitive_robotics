import numpy as np
from rrt import RRT, Node

class CostNode(Node):
    def __init__(self, state):
        super().__init__(state)
        self.cost = 0

class RRT_star(RRT):
    def __init__(self, start_state, goal_state, bounds, obstacles, obstacle_coords = None, gamma=1):
        super().__init__(start_state, goal_state, bounds, obstacles, obstacle_coords)
        self.gamma = gamma
        self.d = self.start_state.shape[0]
        # correct internal data to use CostNode
        self.T.remove_node(self.start_node)
        self.start_node = CostNode(start_state)
        self.T.add_node(self.start_node)

    # adds new_node to the tree and adds an edge from tree_node to new_node
    def extend(self, tree_node, new_node):
        super().extend(tree_node, new_node)
        distance = self.distance(tree_node.state, new_node.state)
        new_node.cost = tree_node.cost + distance

    # returns a set of nodes which are near target_state
    def near(self, target_state):
        n = self.T.number_of_nodes() + 1 # avoid fail when |V| == 1
        max_dist_for_near = self.gamma * (np.log(n) / n) ** (1 / self.d)
        S_near = set()
        for tree_node in self.T.nodes:
            distance = self.distance(tree_node.state, target_state)
            if distance <= max_dist_for_near:
                S_near.add(tree_node)
        return S_near

    # chooses the closest node to x_new in N_near
    def choose_parent(self, x_new, N_near):
        n_min = None
        min_val = np.infty
        for n_near in N_near:
            val = n_near.cost + self.distance(n_near.state, x_new)
            if val < min_val:
                min_val = val
                n_min = n_near
        return n_min

    # n_new becomes parent of any nodes in N_near
    #     that can be reached with lower cost from n_new
    def rewire(self, n_new, N_near):
        for n_near in N_near:
            val = n_new.cost + self.distance(n_new.state, n_near.state)
            cheaper = val < n_near.cost
            obstructed = self.through_obstacle(n_new.state, n_near.state)
            if cheaper and not obstructed:
                n_parent = list(self.T.predecessors(n_near))[0]
                self.T.remove_edge(n_parent, n_near)
                self.T.add_edge(n_new, n_near)
                n_near.cost = val

    # run RRT* for num_iters
    def run(self, num_iters):
        for i in range(num_iters):
            x_rand = self.random()
            n_nearest = self.nearest(x_rand)
            x_new = self.steer(n_nearest.state, x_rand)
            N_near = self.near(x_new)
            n_min = self.choose_parent(x_new, N_near)
            if n_min and not self.through_obstacle(n_min.state, x_new):
                n_new = CostNode(x_new)
                self.extend(n_min, n_new)
                self.rewire(n_new, N_near - set([n_min]))

