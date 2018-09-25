import numpy as np
import control
from rrt_star import RRT_star

# see spec for RRT_star
# A_fn and B_fn take state, control and return matrix
# Q and R are matrices
# update_fn takes state, control and returns a new state
class LQR_RRT_star(RRT_star):
    def __init__(self, s_init, s_goal, bounds, obstacles, A_fn, B_fn, Q, R, update_fn, obstacle_coords = None, gamma = 1):
        super().__init__(s_init, s_goal, bounds, obstacles, obstacle_coords, gamma)
        self.A_fn = A_fn
        self.B_fn = B_fn
        self.Q = Q
        self.R = R
        self.update_fn = update_fn
        self.lqr_memo = {}

    def lqr(self, x_0, u_0):
        lqr_memo_key = (tuple(x_0), tuple(u_0))
        if lqr_memo_key in self.lqr_memo:
            return self.lqr_memo[lqr_memo_key]
        A = self.A_fn(x_0, u_0)
        B = self.B_fn(x_0, u_0)
        K, S, E = control.lqr(A, B, self.Q, self.R)
        self.lqr_memo[lqr_memo_key] = (K, S)
        return (K, S)

    # linearization should be done around the to state
    # TODO: should we ever consider nonzero control?
    # TODO: correct control dimensions (don't assume 1)

    def distance(self, from_state, to_state):
        K, S = self.lqr(to_state, np.zeros(1))
        x_bar = from_state - to_state
        return np.dot(np.dot(x_bar.T, S), x_bar)

    def steer(self, from_state, to_state):
        K, S = self.lqr(to_state, np.zeros(1))
        x_bar = from_state - to_state
        u = - np.dot(K, x_bar)
        return self.update_fn(from_state, u)
