import math
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
from nose.tools import assert_equal, assert_true, ok_
import numpy as np
from numpy import linalg as LA
from IPython.display import display, HTML, clear_output
import utils


def test_ok():
    """ If execution gets to this point, print out a happy message """
    try:
        from IPython.display import display_html
        display_html("""<div class="alert alert-success">
        <strong>Tests passed!!</strong>
        </div>""", raw=True)
    except:
        print("Tests passed!!")


def test_correct_A_matrix(student_A_fn):
    # X and U are 1-dimensional matrices (essentially a python list)
    # i.e. X = [x_value, y_value, theta_value], U = [v_value, phi_value]
    X_1 = [32, 51, 4.19]
    U_1 = [1.75, 1.48]
    solution_1 = np.array([[0, 0, 1.52], [0, 0, -0.87], [0, 0, 0]])

    X_2 = [2, 1, .9]
    U_2 = [-1, 5.2]
    solution_2 = np.array([[0, 0, 0.78], [0, 0, -0.62], [0, 0, 0]])

    X_3 = [590, 128, 3.25]
    U_3 = [5.23, 1.24]
    solution_3 = np.array([[0, 0, 0.57], [0, 0, -5.2], [0, 0, 0]])

    student_solution_1 = student_A_fn(X_1, U_1)

    assert_true(np.array_equal(student_A_fn(X_1, U_1), solution_1), msg="Incorrect A matrix returned")
    assert_true(np.array_equal(student_A_fn(X_2, U_2), solution_2), msg="Incorrect A matrix returned")
    assert_true(np.array_equal(student_A_fn(X_3, U_3), solution_3), msg="Incorrect A matrix returned")

    test_ok()


def test_correct_B_matrix(student_B_fn):
    # X and U are 1-dimensional matrices (essentially a python list)
    # i.e. X = [x_value, y_value, theta_value], U = [v_value, phi_value]
    X_1 = [32, 51, 4.19]
    U_1 = [1.75, 1.48]
    solution_1 = np.array([[-0.5, 0], [-0.87, 0], [10.98, 212.86]])

    X_2 = [2, 1, .9]
    U_2 = [-1, 5.2]
    solution_2 = np.array([[0.62, 0], [0.78, 0], [-1.89, -4.56]])

    X_3 = [590, 128, 3.25]
    U_3 = [5.23, 1.24]
    solution_3 = np.array([[-0.99, 0], [-0.11, 0], [2.91, 49.58]])

    assert_true(np.array_equal(student_B_fn(X_1, U_1), solution_1), msg="Incorrect B matrix returned")
    assert_true(np.array_equal(student_B_fn(X_2, U_2), solution_2), msg="Incorrect B matrix returned")
    assert_true(np.array_equal(student_B_fn(X_3, U_3), solution_3), msg="Incorrect B matrix returned")

    test_ok()


def test_euclidean_cost(student_fn):
    # X_p and X_g are 1-dimensional matrices (essentially a python list)
    X_g = [21, 43, 2.34]
    X_p_1 = [0, 0, 0]
    X_p_2 = [3, 10, 3.14]
    X_p_3 = [12, 98, 1.43]
    X_p_4 = [47, 12, 2.1]
    X_p_5 = [49, 62, 5.79]
    X_p_6 = [93, 43, 2.99]
    solutions = [47.91, 37.60, 55.74, 40.46, 34.01, 72, 0]

    assert_equal(student_fn(X_p_1, X_g), solutions[0], msg="Incorrect cost returned")
    assert_equal(student_fn(X_p_2, X_g), solutions[1], msg="Incorrect cost returned")
    assert_equal(student_fn(X_p_3, X_g), solutions[2], msg="Incorrect cost returned")
    assert_equal(student_fn(X_p_4, X_g), solutions[3], msg="Incorrect cost returned")
    assert_equal(student_fn(X_p_5, X_g), solutions[4], msg="Incorrect cost returned")
    assert_equal(student_fn(X_p_6, X_g), solutions[5], msg="Incorrect cost returned")
    assert_equal(student_fn(X_g, X_g), solutions[6], msg="Incorrect cost returned")

    test_ok()


def test_cost_functions(euclidean_distance):
    X_g = [0, 0, 3.14]
    X_p = [0, 0, 0.001]
    U = [1, 0.001]

    euclidean_cost = euclidean_distance(X_p, X_g) # 3.14

    A = np.array([[0, 0, 0.0], [0, 0, 1.0], [0, 0, 0]])
    B = np.array([[1.0, 0], [0.0, 0], [0.0, 1.0]])
    K, S, E = utils.lqr(A, B)
    lqr_cost = np.matmul(np.matmul(np.subtract(X_g, X_p), S), np.subtract(X_g, X_p))
    lqr_cost = round(np.array(lqr_cost)[0][0], 4)

    print("Euclidean Cost: ", euclidean_cost)
    print("LQR Cost: ", lqr_cost)


def test_ok():
    """ If execution gets to this point, print out a happy message """
    try:
        from IPython.display import display_html
        display_html("""<div class="alert alert-success">
        <strong>Tests passed!!</strong>
        </div>""", raw=True)
    except:
        print("Tests passed!!")

def lqr_steer_test(student_func):
    X_s = [0, 0, math.pi/4]
    X_g = [3, 0, -math.pi/4]
    U_g = [0.1, 0.001]

    correct_cost = 58.90102842

    traj_student, cost_student = student_func(X_s, X_g, U_g)

    traj_student_array = np.array(traj_student)
    plt.figure()
    plt.plot(traj_student_array[:, 0], traj_student_array[:, 1], 'b')
    plt.plot(0, 0, 'bo')
    plt.plot(3, 0, 'ro')
    plt.show()

    print("Blue point is start configuration")
    print("Red point is goal configuration")

    # If the following condition is true, test should fail
    if abs(cost_student - correct_cost)>0.1*correct_cost or LA.norm(np.subtract(traj_student[len(traj_student)-1], X_g))>0.3:
        assert_equal(1,2,msg="Incorrect LQR Steer")

    test_ok()


def rewire_test(student_func):
    goal_tree=nx.DiGraph()
    goal_tree.add_node('S',X=[0,0,0])
    goal_tree.add_node('A',X=[4,3,0])
    goal_tree.add_node('B',X=[5,13,0])
    goal_tree.add_node('C',X=[10,1,0])
    goal_tree.add_node('D',X=[8,6,0])

    goal_tree.add_edge('S','A',cumulative_cost=5)
    goal_tree.add_edge('S','B',cumulative_cost=13)
    goal_tree.add_edge('A','D',cumulative_cost=10)
    goal_tree.add_edge('D','C',cumulative_cost=15.39)

    student_tree = nx.DiGraph()
    student_tree.add_node('S',X=[0,0,0])
    student_tree.add_node('A',X=[4,3,0])
    student_tree.add_node('B',X=[5,13,0])
    student_tree.add_node('C',X=[10,1,0])
    student_tree.add_node('D',X=[8,6,0])

    student_tree.add_edge('S','A',cumulative_cost=5)
    student_tree.add_edge('A','D',cumulative_cost=10)
    student_tree.add_edge('S','B',cumulative_cost=13)
    student_tree.add_edge('B','C',cumulative_cost=26)



    #x_new=student_tree.nodes["D"]
    #x_near_set.append(student_tree.nodes['C'])
    student_func(student_tree,{"C"},'D') #this modifies student_tree


    if len(student_tree.adj)!=len(goal_tree.adj):
        print('Incorrect')
        return None


    for node in student_tree.adj:
        for node_goal in goal_tree.adj:
            if node==node_goal:
                #print(node,node_goal)
                #print(student_tree.adj[node],goal_tree.adj[node_goal])
                #print(student_tree.adj[node]==goal_tree.adj[node_goal])
                if student_tree.adj[node]!=goal_tree.adj[node_goal]:
                    print('Incorrect')
                    return None

    test_ok()
