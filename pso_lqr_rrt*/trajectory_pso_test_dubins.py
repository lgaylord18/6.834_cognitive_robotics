import matplotlib
import matplotlib.pyplot as plt
from trajectory_pso import Trajectory_PSO
from dubins import *

pso = Trajectory_PSO(num_coordinates, num_waypoints, num_trajectories, traj_cost_fn, start, end)
cost, traj = pso.run(100)

# plot
fig = plt.figure()
ax = fig.gca()

# trajectory
x = [p[0] for p in traj]
y = [p[1] for p in traj]
ax.plot(x, y, marker='o')

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

ax.set_title("PSO Trajectory Dubins")
plt.show()
