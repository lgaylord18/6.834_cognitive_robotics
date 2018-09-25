import numpy as np

start = np.array([0, 0, 0])
end = np.array([1, 1, 0])
state_bounds = np.array([[-1, -1, -np.pi], [2, 2, np.pi]])
max_speed = 2.5
max_turn = np.pi / 3
control_bounds = np.array([[0, -max_turn], [max_speed, max_turn]])
obstacles = np.array([[.4, .4, .2], [.95, .7, .15]])
obstacle_coords = [0, 1]
v = 1
phi = .001
L = .4
def A_fn(x, u):
    return np.array([
        [0, 0, -v * np.sin(x[2])],
        [0, 0, v * np.cos(x[2])],
        [0, 0, 0]
    ])
def B_fn(x, u):
    return np.array([
        [np.cos(x[2]), 0],
        [np.sin(x[2]), 0],
        [1/L*np.tan(phi), v/L/(np.cos(phi))**2]
    ])
Q = np.diag([1, 1, 1])
R = np.diag([1, 1])
dt = 0.05
def pi_2_pi(angle):
    while(angle > np.pi):
        angle = angle - 2.0 * np.pi
    while(angle <= -np.pi):
        angle = angle + 2.0 * np.pi
    return angle
def update_fn(x, u):
    u = np.clip(u, *control_bounds)
    return np.array([
      max(x[0] + dt * u[0] * np.cos(x[2]), 0),
      max(x[1] + dt * u[0] * np.sin(x[2]), 0),
      pi_2_pi(x[2] + dt * u[0] / L * np.tan(u[1]))
    ])

def round(x):
    return np.rint(x).astype(np.int)
num_waypoints = 2 * round((np.linalg.norm(start - end) / max_speed) / dt)
print("num_waypoints:", num_waypoints)
num_trajectories = 256
num_coordinates = start.shape[0]

penalty_constant = 3
equality_tolerance = .01
def approx_eq(a, b):
    return np.absolute(a - b) < equality_tolerance

def cost_distance(traj):
    points = traj[:,:-1]
    distances = np.linalg.norm(points[1:] - points[:-1], axis=1)
    length_actual = np.sum(distances)
    length_line = np.linalg.norm(start - end)
    return abs(1 - length_line / length_actual)

# http://mathworld.wolfram.com/Circle-LineIntersection.html
# not 100% sure this works in 3d but that doesn't matter
def count_collisions(w, o):
    num_collisions = 0
    a = w[:-1, obstacle_coords]
    b = w[1:, obstacle_coords]
    c = o[:,:-1]
    r2 = o[:,-1]**2
    d2 = np.sum(np.abs(b - a)**2, axis=-1) # norm squared
    for i in range(d2.shape[0]):
        x = np.cross(-c + a[i], -c + b[i], axis=1)
        if (len(x.shape) == 1):
            x = x.reshape((-1, 1))
        D = np.linalg.norm(x, axis=1)
        if np.any(d2[i] * r2 - D**2 > 0):
            num_collisions += 1
    return num_collisions


num_divisions = 4
def cost_collisions(traj):
    num_collisions = count_collisions(traj, obstacles)
    penalty = penalty_constant if num_collisions else 0
    return penalty + num_collisions / num_waypoints

def cost_feasable(traj):
    points = traj[:,:-1]
    displacements = points[1:] - points[:-1]
    distances = np.linalg.norm(displacements, axis=1)
    speeds = distances / dt

    angles = traj[:,-1]
    angle_changes = angles[1:] - angles[:-1]
    turns = np.arctan((angle_changes / speeds) / dt * L)

    x_changes = displacements[:,0]
    valid_x_changes = approx_eq(x_changes, dt * speeds * np.cos(angles[:-1]))
    num_invalid_x_changes = np.sum(np.logical_not(valid_x_changes))

    y_changes = displacements[:,1]
    valid_y_changes = approx_eq(y_changes, dt * speeds * np.sin(angles[:-1]))
    num_invalid_y_changes = np.sum(np.logical_not(valid_y_changes))

    si = np.any(speeds > max_speed)
    ti = np.any(turns > max_turn)
    xi = num_invalid_x_changes > 0
    yi = num_invalid_y_changes > 0

    infeasable = si or ti or xi or yi
    penalty = penalty_constant if infeasable else 0

    speed_cost = np.sum(speeds / max_speed) / num_waypoints
    turn_cost = np.sum(turns / max_turn) / num_waypoints
    x_cost = num_invalid_x_changes / num_waypoints
    y_cost = num_invalid_y_changes / num_waypoints

    return penalty + .25 * (speed_cost + turn_cost + x_cost + y_cost)

def traj_cost_fn(traj):
    return cost_distance(traj) + cost_collisions(traj) + cost_feasable(traj)
