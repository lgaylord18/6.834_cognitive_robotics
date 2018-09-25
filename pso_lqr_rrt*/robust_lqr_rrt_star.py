"""
DOCUMENTATION:
This file contains the code for a robust version of LQR-RRT*. This extension is based off of
the Luders10_GNC.pdf file included in the directory, specifically section B.

The major differences between this version and our Grand Challenge implementation version are
that this is able to use chance constraints to allocate risk for avoiding obstacles and
it is able to handle obstacles of any polygon shape. Obstacles are defined as a set of linear
equations of the form Ax <= b and determines the robots chances of collision by comparing the
robot position to this equations. An example of the map can be found below. The overall risk
limit is specified and the program ensures the robot does not exceed this risk.
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
delta = 0.1

L = 1
v = 1
phi = 0.001

Q = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 1]])
R = np.array([[1, 0], [0, 1]])
N = np.zeros((3, 2))

initial_configuration = np.array([1, 1, 0.001])
initial_cov = 0.01*np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 0.000001, 0, 0, 0],
                        [0, 0, 0, 0.0001, 0, 0], [0, 0, 0, 0, 0.0001, 0], [0, 0, 0, 0, 0, 0.000001]])
n_tree = 10
goal_point = [4, 1]

map_1 = {
    'obstacles': [
        {
            'A': np.array([
                [-1, 0],
                [1, 0],
                [0, -1],
                [0, 1]
            ]),
            'b': np.array([-2.9, 3.1, 0, 0.9])
        },
        {
            'A': np.array([
                [-1, 0],
                [1, 0],
                [0, -1],
                [0, 1]
            ]),
            'b': np.array([-2.9, 3.1, -1.1, 2])
        },
    ],
    'x': 7,
    'y': 7
}

def lqr_rrt_star(map_1, initial_configuration, n_tree, goal_point):

    G = nx.DiGraph()
    G.add_node(0, configuration=initial_configuration, cumulative_cost=0, covariance=initial_cov, cum_del=0)
    i=0

    while i<n_tree:
        x_rand = sample_new_configuration(map_1, goal_point)
        x_nearest = lqr_nearest(G, x_rand)
        while x_nearest==None:
            x_rand = sample_new_configuration(map_1, goal_point)
            x_nearest = lqr_nearest(G, x_rand)
        x_new_temp, traj_new, steer_ind = lqr_steer(x_nearest, x_rand)
        while x_nearest==None or len(traj_new['points'])==0:
            x_rand = sample_new_configuration(map_1, goal_point)
            x_nearest = lqr_nearest(G, x_rand)
            x_new_temp, traj_new, steer_ind = lqr_steer(x_nearest, x_rand)
        x_new = {'configuration': traj_new['points'][len(traj_new['points'])-1], 'cumulative_cost': x_nearest['cumulative_cost']+traj_new['cost'],
                 'covariance': traj_new['cov'][len(traj_new['cov'])-1]}
        x_near_ind_set = lqr_near(G, x_new)
        x_min, traj_min, min_ind = choose_parent(G, x_near_ind_set, x_new)
        if x_min!=None and collision_free(map_1, traj_min, delta):
            G.add_node(i+1, configuration=x_new['configuration'], cumulative_cost=x_min['cumulative_cost']+traj_min['cost'], covariance=x_new['covariance'])
            G.add_edge(min_ind, i+1, points=traj_min['points'], cost=traj_min['cost'])
            tree = rewire(G, x_near_ind_set, x_new, i+1)
            i=i+1
            print(i)

    return G, find_path(G, goal_point)

def lqr_nearest(tree, x_rand):

    min_cost = np.infty
    min_node = None

    for node in tree.nodes:
        if LA.norm(np.subtract(tree.nodes[node]['configuration'], x_rand['configuration']))<dist_p_1:
            x_temp, traj_temp, steer_ind = lqr_steer(tree.nodes[node], x_rand)
            if traj_temp['cost']<min_cost and collision_free(map_1, traj_temp, delta) and steer_ind:
                min_cost = traj_temp['cost']
                min_node = tree.nodes[node]

    return min_node

def lqr_near(tree, x_new):

    near_node_cost = np.infty
    near_node_set = []

    for node in tree.nodes:
        if LA.norm(np.subtract(tree.nodes[node]['configuration'], x_new['configuration']))<dist_p_1:
            x_temp, traj_temp, steer_ind = lqr_steer(tree.nodes[node], x_new)
            if traj_temp['cost']<near_node_cost and collision_free(map_1, traj_temp, delta) and steer_ind:
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

    # robust
    P_pre = x_s['covariance']
    covariance_list = [initial_cov]

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

        # robust
        A_r = np.add(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), dt*np.array([[0, 0, -u[0]*math.sin(x_temp[2])], [0, 0, u[0]*math.cos(x_temp[2])], [0, 0, 0]]))
        B_r = dt*np.array([[math.cos(x_temp[2]), 0], [math.sin(x_temp[2]), 0], [1/L*math.tan(u[1]), v/L/(math.cos(u[1]))**2]])
        C_r = np.array([[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]])
        D_r = np.array([[1, 0], [0, 1]])

        A_r_mat = np.matrix(A_r)
        B_r_mat = np.matrix(B_r)
        C_r_mat = np.matrix(C_r)
        D_r_mat = np.matrix(D_r)
        S_r_mat = np.matrix(scipy.linalg.solve_discrete_are(A_r_mat, B_r_mat, C_r_mat, D_r_mat))
        S_r = np.array(S_r_mat)
        L_r = np.array(np.matrix(scipy.linalg.inv(B_r_mat.T*S_r_mat*B_r_mat+D_r_mat)*(B_r_mat.T*S_r_mat*A_r_mat)))

        # L_r, S_r, E_r = control.dlqr(A_r, B_r, C_r, D_r, N_r)
        H_r = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        V_r = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        W_r = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        M_r = np.array([[0.000001, 0, 0], [0, 0.000001, 0], [0, 0, 0.00000001]])
        N_r = np.array([[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.0001]])
        P_pre_small = P_pre[:3, :3]
        P_minus = np.add(np.matmul(np.matmul(A_r, P_pre_small), np.transpose(A_r)), np.matmul(np.matmul(V_r, M_r), np.transpose(V_r)))
        temp_mat = np.add(np.matmul(np.matmul(H_r, P_minus), np.transpose(H_r)), np.matmul(np.matmul(W_r, N_r), np.transpose(W_r)))
        if ind>1000:
            print(P_pre)
            print(temp_mat)
        K_r = np.matmul(np.matmul(P_minus, np.transpose(H_r)), LA.inv(temp_mat))
        A_1 = A_r
        A_2 = np.matmul(B_r, L_r)
        A_3 = np.matmul(np.matmul(K_r, H_r), A_r)
        A_4 = np.subtract(np.add(A_r, np.matmul(B_r, L_r)), np.matmul(np.matmul(K_r, H_r), A_r))
        A_total = np.r_[np.c_[A_1, A_2], np.c_[A_3, A_4]]
        V_1 = V_r
        V_2 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        V_3 = np.matmul(np.matmul(K_r, H_r), V_r)
        V_4 = np.matmul(K_r, W_r)
        V_total = np.r_[np.c_[V_1, V_2], np.c_[V_3, V_4]]
        M_1 = M_r
        M_2 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        M_3 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        M_4 = N_r
        M_total = np.r_[np.c_[M_1, M_2], np.c_[M_3, M_4]]
        if ind==-1:
            print(u)
            print(A_total)
            print(np.matmul(np.matmul(A_total, covariance_list[0]), np.transpose(A_total)))
            print(V_total)
        P_temp = np.add(np.matmul(np.matmul(A_total, P_pre), np.transpose(A_total)), np.matmul(np.matmul(V_total, M_total), np.transpose(V_total)))
        covariance_list.append(P_temp)
        P_pre = P_temp
        covariance_list.append(initial_cov)
        P_pre = initial_cov

    if dist >= max_dist:
        steer_ind = False

    x = {'configuration': x_1, 'cumulative_cost': x_s['cumulative_cost']+traj_cost, 'covariance': covariance_list[len(covariance_list)-1]}
    traj = {'points': x_list, 'cost': traj_cost, 'cov': covariance_list}

    return x, traj, steer_ind

def sample_new_configuration_uniform(map_1, goal_point):

    x = random.uniform(0, map_1['x'])
    y = random.uniform(0, map_1['y'])
    conf = [x, y]
    temp_cov = 0.000001*np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])

    while not(collision_free_conf(map_1, conf, temp_cov, 0.5)):
        x = random.uniform(0, map_1['x'])
        y = random.uniform(0, map_1['y'])
        conf = [x, y]

    theta = random.uniform(-np.pi, np.pi)
    x_rand = {'configuration': [x, y, theta], 'cumulative_cost': np.infty}

    return x_rand

def sample_new_configuration_biased(map_1, goal_point):

    del_distance = 0.01
    x = random.uniform(goal_point[0]-del_distance, goal_point[0]+del_distance)
    y = random.uniform(goal_point[1]-del_distance, goal_point[1]+del_distance)
    conf = [x, y]
    temp_cov = 0.000001*np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
    # print(conf)

    while not(collision_free_conf(map_1, conf, temp_cov, 0.5)):
        x = random.uniform(goal_point[0]-del_distance, goal_point[0]+del_distance)
        y = random.uniform(goal_point[1]-del_distance, goal_point[1]+del_distance)
        conf = [x, y]
        # print(conf)

    theta = random.uniform(-np.pi, np.pi)
    x_rand = {'configuration': [x, y, theta], 'cumulative_cost': np.infty}

    return x_rand

def sample_new_configuration(map_1, goal_point):

    if random.random()<0.3:
        return sample_new_configuration_biased(map_1, goal_point)
    else:
        return sample_new_configuration_uniform(map_1, goal_point)

def collision_free_conf(map_1, X_mean, X_covariance, delta):
    '''
        map_1: {
            'obstacles': [
                # Obstacle 1
                {
                    # n is the number of sides on the polygon (i.e. rectangle has 4)
                    # Used to represent Ax < b
                    'A': np.array([
                        [1, 0],
                        [-1, 0],
                        [0, 1],
                        [0, -1]
                    ]), # size: n x 2
                    'b': np.array([0, -1, 0, -1]) # size: 1 x n
                },
                # Obstacle 2
                {
                    'A': [[]],
                    'b': []
                }
            ],
            'x': max_x,
            'y': max_y
        }
        X_mean: np.array([X, Y]) # The mean of all possible robot positions. Size 1 x 2
        X_covariance: np.array([]) # Covariance to represent possible robot positions. Size 6 x 6
        delta: float # Risk allocation for a time step

    '''
    collision_ind = True
    obstacles = map_1['obstacles']

    # These represent the four edges of the available space.
    A_map_1 = np.array([-1, 0])
    b_map_1 = np.array([0])
    A_map_2 = np.array([1, 0])
    b_map_2 = np.array([map_1['x']]) # 5
    A_map_3 = np.array([0, -1])
    b_map_3 = np.array([0])
    A_map_4 = np.array([0, 1])
    b_map_4 = np.array([map_1['y']]) # 5

    map_boundary_eqns = [(A_map_1, b_map_1), (A_map_2, b_map_2), (A_map_3, b_map_3), (A_map_4, b_map_4)]

    X_mean = np.array(X_mean)
    X_covariance = np.array(X_covariance)

    del_i0t = []
    for A_map, b_map in map_boundary_eqns:
        # probability of X_mean making collision with i-th face of map's boundary
        # at time step t.
        del_i0t.append(1/2 * (1 - math.erf((b_map - np.matmul(A_map, X_mean.T))/np.sqrt(2*(np.matmul(np.matmul(A_map, X_covariance[:2, :2]), A_map.T))))));
    del_i0t = np.array(del_i0t)
    # print("del_i0t: ", del_i0t)

    del_ijt=[]
    for i in range(len(obstacles)):
        A = obstacles[i]['A']
        b = obstacles[i]['b']
        # probability that the robot collides with jth face of ith obstacle at
        # time step t.
        del_ijt.append(1/2 * (1 - scipy.special.erf((np.matmul(A, X_mean.T) - b)/np.sqrt(2*(np.matmul(np.matmul(A, X_covariance[:2, :2]), A.T)).diagonal()))));
    del_ijt = np.array(del_ijt)
    # print("del_ijt: ", del_ijt)

    # probability of X_mean colliding with all faces of map's boundary at time step t.
    del_0t = del_i0t.sum()
    # print("del_0t: ", del_0t)

    # probability of X_mean colliding with ith obstacle at time step t.
    del_it = []
    for array in del_ijt:
        del_it.append(array.min())
    del_it = np.array(del_it)
    # print("del_it: ", del_it)

    # probability of X_mean colliding with obstacle and map boundary at time step t.
    del_t = del_0t + sum(del_it)
    # print("del_t: ", del_t)

    # if del_t is bigger than threshold, collision.
    # delta is the minimum probability without collision that we wish the
    # robot to maintain.
    if del_t > delta:
        collision_ind = False
    # print("Collision free: ", collision_free)
    return collision_ind

# import time
# start = time.time()
# for i in range(1000):
#     collision_free_conf(map_1, X_mean, X_covariance, .1)
# end = time.time()
# print("Total time: ", end - start)

def collision_free(map_1, traj, delta):

    if len(traj['points'])==0:
        collision_ind = False

    collision_ind = True

    for i in range(len(traj['points'])):
        # We are only taking the X, Y from the total state ([X, Y, theta])
        # since that is all that is necessary to determine a collision
        X_mean = traj['points'][i][:2]
        X_covariance = traj['cov'][i]
        collision_ind = collision_free_conf(map_1, X_mean, X_covariance, delta)

        if not collision_ind:
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

def rewire(tree, x_near_ind_set, x_new, x_new_ind):

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
        if x_new['cumulative_cost']+traj['cost']<x_near['cumulative_cost'] and collision_free(map_1, traj, delta) and steer_ind:
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

lqr_tree, path = lqr_rrt_star(map_1, initial_configuration, n_tree, goal_point)

"""
matplotlib.pyplot.figure(0)
for i in range(0, n_tree):
    for x_parent in lqr_tree.predecessors(i):
        x_1_conf = lqr_tree.nodes[x_parent]['configuration']
        x_1 = [x_1_conf[0], x_1_conf[1]]
        x_2_conf = lqr_tree.nodes[i]['configuration']
        x_2 = [x_2_conf[0], x_2_conf[1]]
        matplotlib.pyplot.plot([x_1[0], x_2[0]], [x_1[1], x_2[1]])
matplotlib.pyplot.figure(1)
for i in range(0, n_tree):
    for x_parent in lqr_tree.predecessors(i):
        x_list_array = np.array(lqr_tree.edges[x_parent, i]['points'])
        x_node = lqr_tree.nodes[i]['configuration']
        matplotlib.pyplot.plot(x_list_array[:, 0], x_list_array[:, 1])
        matplotlib.pyplot.plot(x_node[0], x_node[1], 'bo')
"""

path_plot_x = []
path_plot_y = []
matplotlib.pyplot.figure(0)
for i in range(0, len(path)):
    path_plot_x.append(path[i][0])
    path_plot_y.append(path[i][1])
matplotlib.pyplot.plot(initial_configuration[0], initial_configuration[1], 'bo')
matplotlib.pyplot.fill([2.9, 3.1, 3.1, 2.9], [0, 0, 0.9, 0.9], 'k')
matplotlib.pyplot.fill([2.9, 3.1, 3.1, 2.9], [1.1, 1.1, 2, 2], 'k')
matplotlib.pyplot.fill([3.5, 4.5, 4.5, 3.5], [0.5, 0.5, 1.5, 1.5], 'b', alpha=0.3)
matplotlib.pyplot.axis([0, 7, 0, 7])
matplotlib.pyplot.plot(path_plot_x, path_plot_y, 'r')
matplotlib.pyplot.show()
