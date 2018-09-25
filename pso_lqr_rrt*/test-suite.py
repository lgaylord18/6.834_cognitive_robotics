import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
import time
from dubins import *
from pso_lqr_rrt_star import PSO_LQR_RRT_star
from lqr_rrt_star import LQR_RRT_star
from trajectory_pso import Trajectory_PSO


num_pso_iterations = 500
num_rrt_iterations = 5000

def test_algs(numRuns):

    pso_routes = []
    lqr_routes = []
    combined_routes = []

    pso_times = []
    lqr_times = []
    combined_times = []

    for i in range(numRuns):
        print("run {} of {}".format(i+1, numRuns))
        # #PSO runs
        start_time = time.time()
        pso = Trajectory_PSO(num_coordinates, num_waypoints, num_trajectories, traj_cost_fn, start, end)
        cost, traj = pso.run(num_pso_iterations)

        dt = time.time()-start_time
        pso_times.append(dt)
        pso_routes.append(traj)
        print(dt)

        # LQR runs
        start_time = time.time()
        alg = LQR_RRT_star(start, end, state_bounds, obstacles, A_fn, B_fn, Q, R, update_fn, obstacle_coords)
        alg.run(num_rrt_iterations)
        traj = alg.best_path(end, .5)

        dt = time.time()-start_time
        lqr_times.append(dt)
        lqr_routes.append(traj)
        print(dt)

        #combined runs
        start_time = time.time()
        alg = PSO_LQR_RRT_star(start, end, state_bounds, obstacles, A_fn, B_fn, Q, R, update_fn, num_waypoints, num_trajectories, traj_cost_fn)
        traj = alg.run(round(.5 * num_pso_iterations), round(.5 * num_rrt_iterations))

        dt = time.time()-start_time
        combined_times.append(dt)
        combined_routes.append(traj)
        print(dt)

    print_routes(pso_routes, 'PSO route map')
    print_routes(lqr_routes, 'LQR route map')
    print_routes(combined_routes, 'Combined Algorithm route map')
    print_times(pso_times,lqr_times,combined_times)
    plt.show()



def print_routes(routes, title):
    
    fig = plt.figure()
    ax = fig.gca()

    num_routes = len(routes)

    ax.set_prop_cycle(cycler('color', [plt.cm.Spectral(i) for i in np.linspace(0, 0.9, num_routes)]))

    # path
    for route in routes:
        x = [p[0] for p in route]
        y = [p[1] for p in route]
        ax.plot(x,y)

    # start & goal
    start_coords = start[:2].tolist()
    goal_coords = end[:2].tolist()
    ax.scatter(*start_coords, color='b')
    ax.scatter(*goal_coords, color='b')

  # obstacles
    for obstacle in obstacles:
        radius = obstacle[-1]
        center = obstacle[:-1]
        ax.add_patch(plt.Circle(center, radius, color='r'))

    ax.set_title(title)

def print_times(pso, lqr, combined):
    fig = plt.figure()
    ax = fig.gca()

    labels = ["pso", "LQR-RRT*", "Combined Algorithm"]
    ax.plot(pso, linestyle='--', linewidth=1)
    ax.plot(lqr,linestyle='-.', linewidth=2)
    ax.plot(combined, linestyle=':', linewidth=3)

    ax.legend(labels)

    ax.set_title("Timing Diagram for the 3 Algorithms (sec)")



def test_plot():
    routes = []

    start = (0,0)

    end = (20,20)

    for t in np.arange(0,1,.01):
        route = [start]
        xs = np.linspace(start[0],end[0]*np.sin(t*np.pi/2), num=20)
        ys = np.linspace(start[1],end[1]*np.cos(t*np.pi/2), num=20)
        route = [(xs[i],ys[i]) for i in range(len(xs))]
        routes.append(route)

    print_routes(routes, "Test")

    print_times(np.linspace(1,5,num=50),np.linspace(2,3,num=50),np.linspace(5,1,num=50))

    plt.show()

test_algs(5)
