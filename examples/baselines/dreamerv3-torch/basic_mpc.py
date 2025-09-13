from pydrake.all import (MathematicalProgram, Solve, ClarabelSolver)
import numpy as np

class EndEffectorMPC:
    def __init__(self, goal_pos, horizon, Q = np.eye(3), R = 1e-2*np.eye(3), env_id=0, random=True):
        self.Q = Q
        self.R = R
        self.horizon = horizon
        self.env_id = env_id
        A = np.eye(3)
        B = np.diag([0.04, 0.015, 0.0175]) # Approximate delta rates

        # Define all the randomness to get different policies
        if random:
            self.noise_level = np.random.choice([0, 0.01, 0.02, 0.03, 0.06])
            if np.random.rand() < 0.75:
                goal_pos += np.array([np.random.choice([0, 0.025]), np.random.choice([-0.02, -0.01, 0.0, 0.01, 0.02]), np.random.choice([0.095, 0.1, 0.105])])
                self.return_pos = np.array([0, np.random.choice([-0.2, -0.1, 0, 0.1, 0.2]), 0.4])
                self.R_return = np.diag(10**np.random.uniform(-2, 0, size=3))
            else:
                goal_pos += np.array([0, 0, 0.1])
                self.return_pos = np.array([0, np.random.choice([-0.3, -0.2, 0.2, 0.3]), 0.4])
            self.R_return = np.diag(10**np.random.uniform(-3, -1, size=3))
            self.R = np.diag(10**np.random.uniform(-3, -1, size=3))
            self.time_to_grip = np.random.choice(range(5, 11))
        else:
            goal_pos += np.array([0, 0, 0.1])
            self.noise_level = 25e-2
            self.return_pos = np.array([0, 0, 0.4])
            self.R_return = 1e-1*np.eye(3)
            self.time_to_grip = 10

        # Program and variables
        self.prog = MathematicalProgram()
        self.q_sym = [self.prog.NewContinuousVariables(3, f"q_{k}") for k in range(horizon + 1)]
        self.u_sym = [self.prog.NewContinuousVariables(3, f"u_{k}") for k in range(horizon)]

        # Dynamics constraint
        for k in range(horizon):
            self.prog.AddConstraint(np.equal(A@self.q_sym[k] + B@self.u_sym[k], self.q_sym[k+1], dtype=object))

        # Initial condition constraint
        self.ic_constraint = self.prog.AddConstraint(np.equal(self.q_sym[k], np.zeros(3), dtype=object))

        # Control limit constraints
        self.prog.AddBoundingBoxConstraint(-1.0, 1.0, np.concatenate(self.u_sym))

        # Tracking costs
        self.add_tracking_cost(goal_pos)

        # Init solver
        self.solver = ClarabelSolver()

        # Dumb state machine
        self.state = 0

    def add_tracking_cost(self, goal_pos):
        # Clear costs
        [self.prog.RemoveCost(c) for c in self.prog.GetAllCosts()]

        self.goal_pos = goal_pos
        for k in range(self.horizon):
            self.prog.AddCost((self.q_sym[k+1] - goal_pos).dot(self.Q@(self.q_sym[k+1] - goal_pos)))
            self.prog.AddCost(self.u_sym[k].dot(self.R@self.u_sym[k]))

    def set_initial_condition(self, q_ic):
        self.prog.RemoveConstraint(self.ic_constraint)
        self.ic_constraint = self.prog.AddConstraint(np.equal(self.q_sym[0], q_ic, dtype=object))
 
    def get_action(self, q_ic, action):
        self.set_initial_condition(q_ic)
        result = self.solver.Solve(self.prog)

        # Dumb state machine
        action['action'][self.env_id, :] = 0
        if self.state == 0 and np.linalg.norm(self.goal_pos - q_ic) >= 1e-2:
            action['action'][self.env_id, :3] = result.GetSolution(self.u_sym[0]) + self.noise_level*np.random.randn(3)
            action['action'][self.env_id, 6] = 0.55
        elif self.state < self.time_to_grip:
            self.state += 1
            action['action'][self.env_id, 6] = 0
        elif self.state == self.time_to_grip:
            self.R = self.R_return
            self.add_tracking_cost(self.return_pos)
            self.state += 1
        else:
            action['action'][self.env_id, :3] = result.GetSolution(self.u_sym[0]) + self.noise_level*np.random.randn(3)
            action['action'][self.env_id, 6] = 0
        return action