"""
DOCUMENTATION:
This files contains the code for the underactuated RRT* that will be used in the Grand Challenge.
It takes a bit map of 1s and 0s as input along with the robots current position and the goal point
to reach. The bit map contains a 1 if there is an obstacle and a zero otherwise. Using this bitmap,
we create our own representation of the world in the format below.
    map: {
        'obstacles': [
            {'center': [3, 1], 'radius': 1},
            {'center': [6, 4], 'radius': 1}
        ],
        'x': [0, 9], (min_x, max_x)
        'y': [0, 5], (min_y, max_y)
    }

After constructing the map, we run a directed LQR-RRT* on the graph to focus on getting our
vehicle from the current position to the goal point (or near it). After sampling points using
LQR-RRT*, we perform a shortest path search to give the points needed to travel to in order
to reach the goal point.

For an example, please refer to the bottom of the file.
"""
import numpy as np
import networkx as nx
from numpy import linalg as LA
import math
import matplotlib
import scipy
import control
import random

dist_p_1 = 100

max_dist = 0.1
max_ind = 500
dt = 0.01

L = 0.1
v = 0.2
phi = 0.001

Q = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 5]])
R = np.array([[1, 0], [0, 1]])
N = np.zeros((3, 2))

MAP_RESOLUTION = 0.05


def lqr_rrt_star(bit_map, initial_configuration, goal_point):
    """
        bit_map (np.array): List of lists with 0s in free space and 1s at obstacles
        initial_configuration (np.array): [X, Y, theta]
        goal_point (np.array): [X, Y]
    """
    n_tree = 20
    # Initial map setup from the bit_map
    map_1 = {}
    map_1['x'] = [0, len(bit_map)*MAP_RESOLUTION]
    map_1['y'] = [0, len(bit_map[0])*MAP_RESOLUTION]
    map_1['obstacles'] = []
    for i in range(len(bit_map)):
        for j in range(len(bit_map[0])):
            if bit_map[i][j] == 1:
                map_1['obstacles'].append({'center': [i*MAP_RESOLUTION, j*MAP_RESOLUTION], 'radius': MAP_RESOLUTION*(np.sqrt(2)/2)})

    G = nx.DiGraph()
    G.add_node(0, configuration=initial_configuration, cumulative_cost=0)
    i=0

    while i<n_tree:
        x_rand = sample_new_configuration(map_1, goal_point)
        x_nearest = lqr_nearest(G, x_rand, map_1)
        while x_nearest==None:
            x_rand = sample_new_configuration(map_1, goal_point)
            x_nearest = lqr_nearest(G, x_rand, map_1)
        x_new_temp, traj_new, steer_ind = lqr_steer(x_nearest, x_rand)
        x_new = {'configuration': traj_new['points'][len(traj_new['points'])-1], 'cumulative_cost': x_nearest['cumulative_cost']+traj_new['cost']}
        x_near_ind_set = lqr_near(G, x_new, map_1)
        x_min, traj_min, min_ind = choose_parent(G, x_near_ind_set, x_new)
        if x_min!=None and collision_free(map_1, traj_min):
            G.add_node(i+1, configuration=x_new['configuration'], cumulative_cost=x_min['cumulative_cost']+traj_min['cost'])
            G.add_edge(min_ind, i+1, points=traj_min['points'], cost=traj_min['cost'])
            tree = rewire(G, x_near_ind_set, x_new, i+1, map_1)
            i=i+1
            print(i)

    # return G
    return G, find_path(G, goal_point)

def lqr_nearest(tree, x_rand, map_1):

    min_cost = np.infty
    min_node = None

    for node in tree.nodes:
        if LA.norm(np.subtract(tree.nodes[node]['configuration'], x_rand['configuration']))<dist_p_1:
            x_temp, traj_temp, steer_ind = lqr_steer(tree.nodes[node], x_rand)
            if traj_temp['cost']<min_cost and collision_free(map_1, traj_temp) and steer_ind:
                min_cost = traj_temp['cost']
                min_node = tree.nodes[node]

    return min_node

def lqr_near(tree, x_new, map_1):

    near_node_cost = np.infty
    near_node_set = []

    for node in tree.nodes:
        if LA.norm(np.subtract(tree.nodes[node]['configuration'], x_new['configuration']))<dist_p_1:
            x_temp, traj_temp, steer_ind = lqr_steer(tree.nodes[node], x_new)
            if traj_temp['cost']<near_node_cost and collision_free(map_1, traj_temp) and steer_ind:
                near_node_set.append(node)

    return near_node_set

def lqr_steer(x_s, x_g):

    """
    inputs
    x_s = {'configuration': conf, "cumulative_cost": cost}
    x_s = {'configuration': conf, "cumulative_cost": cost}

    outputs
    x = {'configuration': conf, 'cumulative_cost': cost}
    traj = {'points': list_of_intermediate_points, 'cost': cost}
    steer_ind = True or False
    """

    x_0 = x_s['configuration']
    x_1 = x_g['configuration']
    theta = x_1[2]
    A = [[0, 0, -v*math.sin(theta)], [0, 0, v*math.cos(theta)], [0, 0, 0]]
    B = [[math.cos(theta), 0], [math.sin(theta), 0], [1/L*math.tan(phi), v/L/(math.cos(phi))**2]]
    K, S, E = control.lqr(A, B, Q, R, N)
    traj_cost = np.matmul(np.matmul(np.subtract(x_1, x_0), S), np.subtract(x_1, x_0))
    ind = 1
    x_list = [x_0]
    x_pre = x_0
    x_norm_1 = [x_pre[0], x_pre[1], 1*x_pre[2]]
    x_norm_2 = [x_1[0], x_1[1], 1*x_1[2]]
    dist = LA.norm(np.subtract(x_norm_1, x_norm_2))
    steer_ind = True

    while dist >= max_dist and ind <= max_ind :
        x_temp = [0, 0, 0]
        u = np.negative(np.matmul(K, np.subtract(x_pre, x_1)))
        x_temp[0] = x_pre[0] + dt*u[0]*math.cos(x_pre[2])
        x_temp[1] = x_pre[1] + dt*u[0]*math.sin(x_pre[2])
        x_temp[2] = x_pre[2] + dt*u[0]/L*math.tan(u[1])
        x_temp[2] = (x_temp[2] + np.pi) % (2 * np.pi ) - np.pi
        x_list.append(x_temp)
        x_pre = x_temp
        x_norm_1 = [x_pre[0], x_pre[1], 1*x_pre[2]]
        x_norm_2 = [x_1[0], x_1[1], 1*x_1[2]]
        dist = LA.norm(np.subtract(x_norm_1, x_norm_2))
        ind = ind + 1

    if dist >= max_dist:
        steer_ind = False

    x = {'configuration': x_1, 'cumulative_cost': x_s['cumulative_cost']+traj_cost}
    traj = {'points': x_list, 'cost': traj_cost}

    return x, traj, steer_ind

def sample_new_configuration_uniform(map_1, goal_point):

    obstacles = map_1['obstacles']
    map_x = map_1['x']
    map_y = map_1['y']

    x = random.uniform(map_x[0], map_x[1])
    y = random.uniform(map_y[0], map_y[1])
    conf = [x, y]

    while not(collision_free_conf(map_1, conf)):
        x = random.uniform(map_x[0], map_x[1])
        y = random.uniform(map_y[0], map_y[1])
        conf = [x, y]

    theta = random.uniform(-np.pi, np.pi)
    x_rand = {'configuration': [x, y, theta], 'cumulative_cost': np.infty}

    return x_rand

def sample_new_configuration_biased(map_1, goal_point):

    x = random.uniform(goal_point[0]-0.5, goal_point[0]+0.5)
    y = random.uniform(goal_point[0]-0.5, goal_point[0]+0.5)
    conf = [x, y]

    while not(collision_free_conf(map_1, conf)):
        x = random.uniform(goal_point[0]-0.5, goal_point[0]+0.5)
        y = random.uniform(goal_point[0]-0.5, goal_point[0]+0.5)
        conf = [x, y]

    theta = random.uniform(-np.pi, np.pi)
    x_rand = {'configuration': [x, y, theta], 'cumulative_cost': np.infty}

    return x_rand

def sample_new_configuration(map_1, goal_point):

    if random.random()<0.3:
        return sample_new_configuration_biased(map_1, goal_point)
    else:
        return sample_new_configuration_uniform(map_1, goal_point)

def collision_free_conf(map_1, conf):

    collision_ind = True
    obstacles = map_1['obstacles']

    for obstacle in obstacles:
        obstacle_center = obstacle['center']
        obstacle_radius = obstacle['radius']
        if LA.norm(np.subtract(conf, obstacle_center))<obstacle_radius:
            collision_ind = False
            break

    return collision_ind

def collision_free(map_1, traj):

    collision_ind = True
    obstacles = map_1['obstacles']
    map_x = map_1['x']
    map_y = map_1['y']

    for x in traj['points']:
        x_conf = [x[0], x[1]]

        if not(x_conf[0]>=map_x[0] and x_conf[0]<=map_x[1] and x_conf[1]>=map_y[0] and x_conf[1]<=map_y[1]):
            collision_ind = False
            break

        for obstacle in obstacles:
            obstacle_center = obstacle['center']
            obstacle_radius = obstacle['radius']
            if LA.norm(np.subtract(x_conf, obstacle_center))<obstacle_radius:
                collision_ind = False
                break

        if not(collision_ind):
            break

    return collision_ind

def choose_parent(tree, x_near_ind_set, x_new):

    """
    Finds the lowest cost parent for the new node from the set of near nodes

    Parameters:
        tree (networkx.DiGraph): The RRT
        x_near_set (list): x_near_set is a list of node keys that are near x_new
        x_new (int): configuration for the new node.

    Returns:
        x_min (networkx.Node): The node that is the minimum cost from x_new
        trajectory_min (networkx.Edge): The edge from x_new to x_min
    """

    min_cost = np.infty
    x_min = None
    trajectory_min = None
    min_ind = None

    # Uses the x_new key to grab the actual node
    for ind in x_near_ind_set:
        x_near = tree.nodes[ind]
        # Steer expects actual nodes, not just node keys
        x, trajectory, steer_ind = lqr_steer(x_near, x_new)
        if x_near['cumulative_cost']+trajectory['cost']<min_cost and steer_ind:
            min_cost = x_near['cumulative_cost'] + trajectory['cost']
            x_min = x_near
            trajectory_min = trajectory
            min_ind = ind

    return x_min, trajectory_min, min_ind

def rewire(tree, x_near_ind_set, x_new, x_new_ind, map_1):

    """
    does local rewiring of the tree. if it finds a better path then it changes the tree to a add this path
    and remove the other worse path

    inputs
    tree
    x_near_ind_set = [list_of_indices_for_near_nodes]
    x_new = {'configuration': conf, "cumulative_cost": cost}
    x_new_ind = index_of_x_new_in_the_tree
    """

    for ind in x_near_ind_set:
        x_near = tree.nodes[ind]
        x, traj, steer_ind = lqr_steer(x_new, x_near)
        if x_new['cumulative_cost']+traj['cost']<x_near['cumulative_cost'] and collision_free(map_1, traj) and steer_ind:
            for x_near_parent in tree.predecessors(ind):
                tree.remove_edge(x_near_parent, ind)
                tree.add_node(ind, configuration=x_near['configuration'], cumulative_cost=x_new['cumulative_cost']+traj['cost'])
                tree.add_edge(x_new_ind, ind, points=traj['points'], cost=traj['cost'])


def euclidean_distance(point1, point2):
    return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def find_path(tree, goal_point):
    """
    Takes in tree and goal_point
    goal_point = [X, Y]

    Outputs list of [x, y, theta] coordinates
    """
    initial_node = tree.nodes[0]
    # Points need to be [X, Y]
    initial_point = [initial_node['configuration'][0], initial_node['configuration'][1]]
    closest_node_to_goal = 0
    closest_distance = euclidean_distance(initial_point, goal_point)

    for node in tree.nodes():
        # point = [X, Y]
        point = [tree.nodes[node]['configuration'][0], tree.nodes[node]['configuration'][1]]
        d = euclidean_distance(point, goal_point)
        if d < closest_distance:
            closest_node_to_goal = node
            closest_distance = d

    # print("tree:", tree)
    # print("init:", tree.nodes[0])
    # print("target:", closest_node_to_goal)
    node_path = nx.shortest_path(tree, 0, closest_node_to_goal)
    path_coords = []
    # Node based
    # for node in node_path:
    #     path_coords.append(tree.nodes[node]['configuration'])

    # Edge based
    for i in range(len(node_path) - 1):
        node_1 = node_path[i]
        node_2 = node_path[i+1]
        edge = tree[node_1][node_2]
        path_coords.extend(edge['points'])

    return path_coords

"""
Below is an example of running our code with visualization.
It specifies the initial configuration of the robot as well as the goal point and a bitmap
with no obstacles.

Beneath that is code to visualize the biased sample points and the final path output by
our algorithm.
"""
# Testing configuration
# Uncomment below in order to run an example.
# initial_configuration = [1, 1, np.pi/2]
# goal_point = [4.5, 4.5]
# bit_map = np.zeros((100, 100))

# Visualization
# lqr_tree, path = lqr_rrt_star(bit_map, initial_configuration, goal_point)
# matplotlib.pyplot.figure(0)
# n_tree = 20
# for i in range(0, n_tree):
#     for x_parent in lqr_tree.predecessors(i):
#         x_1_conf = lqr_tree.nodes[x_parent]['configuration']
#         x_1 = [x_1_conf[0], x_1_conf[1]]
#         x_2_conf = lqr_tree.nodes[i]['configuration']
#         x_2 = [x_2_conf[0], x_2_conf[1]]
#         matplotlib.pyplot.plot([x_1[0], x_2[0]], [x_1[1], x_2[1]])
# matplotlib.pyplot.figure(1)
# for i in range(0, n_tree):
#     for x_parent in lqr_tree.predecessors(i):
#         x_list_array = np.array(lqr_tree.edges[x_parent, i]['points'])
#         x_node = lqr_tree.nodes[i]['configuration']
#         matplotlib.pyplot.plot(x_list_array[:, 0], x_list_array[:, 1])
#         matplotlib.pyplot.plot(x_node[0], x_node[1], 'bo')
# path_plot_x = []
# path_plot_y = []
# for i in range(0, len(path)):
#     path_plot_x.append(path[i][0])
#     path_plot_y.append(path[i][1])
# matplotlib.pyplot.plot(path_plot_x, path_plot_y, 'r')
# matplotlib.pyplot.show()
