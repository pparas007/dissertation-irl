"""
Implements the gridworld MDP.

Matthew Alger, 2015
matthew.alger@anu.edu.au
"""

import numpy as np
import numpy.random as rn

class Gridworld(object):
    """
    Gridworld MDP.
    """

    def __init__(self, grid_size, wind, discount):
        """
        grid_size: Grid size. int.
        wind: Chance of moving randomly. float.
        discount: MDP discount. float.
        -> Gridworld
        """

        self.actions = ((1, 0), (0, 1), (-1, 0), (0, -1))
        self.n_actions = len(self.actions)
        self.n_states = grid_size**2
        self.grid_size = grid_size
        self.wind = wind
        self.discount = discount

        # Preconstruct the transition probability array.
        self.transition_probability = np.array(
            [[[self._transition_probability(i, j, k)
               for k in range(self.n_states)]
              for j in range(self.n_actions)]
             for i in range(self.n_states)])

    def __str__(self):
        return "Gridworld({}, {}, {})".format(self.grid_size, self.wind,
                                              self.discount)

    def feature_vector(self, i, feature_map="ident"):
        """
        Get the feature vector associated with a state integer.

        i: State int.
        feature_map: Which feature map to use (default ident). String in {ident,
            coord, proxi}.
        -> Feature vector.
        """

        if feature_map == "coord":
            f = np.zeros(self.grid_size)
            x, y = i % self.grid_size, i // self.grid_size
            f[x] += 1
            f[y] += 1
            return f
        if feature_map == "proxi":
            f = np.zeros(self.n_states)
            x, y = i % self.grid_size, i // self.grid_size
            for b in range(self.grid_size):
                for a in range(self.grid_size):
                    dist = abs(x - a) + abs(y - b)
                    f[self.point_to_int((a, b))] = dist
            return f
        # Assume identity map.
        f = np.zeros(self.n_states)
        f[i] = 1
        return f

    def feature_matrix(self, feature_map="ident"):
        """
        Get the feature matrix for this gridworld.

        feature_map: Which feature map to use (default ident). String in {ident,
            coord, proxi}.
        -> NumPy array with shape (n_states, d_states).
        """

        features = []
        for n in range(self.n_states):
            f = self.feature_vector(n, feature_map)
            features.append(f)
        return np.array(features)

    def int_to_point(self, i):
        """
        Convert a state int into the corresponding coordinate.

        i: State int.
        -> (x, y) int tuple.
        """

        return (i % self.grid_size, i // self.grid_size)

    def point_to_int(self, p):
        """
        Convert a coordinate into the corresponding state int.

        p: (x, y) tuple.
        -> State int.
        """

        return p[0] + p[1]*self.grid_size

    def neighbouring(self, i, k):
        """
        Get whether two points neighbour each other. Also returns true if they
        are the same point.

        i: (x, y) int tuple.
        k: (x, y) int tuple.
        -> bool.
        """

        return abs(i[0] - k[0]) + abs(i[1] - k[1]) <= 1

    def _transition_probability(self, i, j, k):
        """
        Get the probability of transitioning from state i to state k given
        action j.

        i: State int.
        j: Action int.
        k: State int.
        -> p(s_k | s_i, a_j)
        """

        xi, yi = self.int_to_point(i)
        xj, yj = self.actions[j]
        xk, yk = self.int_to_point(k)

        if not self.neighbouring((xi, yi), (xk, yk)):
            return 0.0

        # Is k the intended state to move to?
        if (xi + xj, yi + yj) == (xk, yk):
            return 1 - self.wind + self.wind/self.n_actions

        # If these are not the same point, then we can move there by wind.
        if (xi, yi) != (xk, yk):
            return self.wind/self.n_actions

        # If these are the same point, we can only move here by either moving
        # off the grid or being blown off the grid. Are we on a corner or not?
        if (xi, yi) in {(0, 0), (self.grid_size-1, self.grid_size-1),
                        (0, self.grid_size-1), (self.grid_size-1, 0)}:
            # Corner.
            # Can move off the edge in two directions.
            # Did we intend to move off the grid?
            if not (0 <= xi + xj < self.grid_size and
                    0 <= yi + yj < self.grid_size):
                # We intended to move off the grid, so we have the regular
                # success chance of staying here plus an extra chance of blowing
                # onto the *other* off-grid square.
                return 1 - self.wind + 2*self.wind/self.n_actions
            else:
                # We can blow off the grid in either direction only by wind.
                return 2*self.wind/self.n_actions
        else:
            # Not a corner. Is it an edge?
            if (xi not in {0, self.grid_size-1} and
                yi not in {0, self.grid_size-1}):
                # Not an edge.
                return 0.0

            # Edge.
            # Can only move off the edge in one direction.
            # Did we intend to move off the grid?
            if not (0 <= xi + xj < self.grid_size and
                    0 <= yi + yj < self.grid_size):
                # We intended to move off the grid, so we have the regular
                # success chance of staying here.
                return 1 - self.wind + self.wind/self.n_actions
            else:
                # We can blow off the grid only by wind.
                return self.wind/self.n_actions

    def reward(self, state_int):
        """
        Reward for being in state state_int.

        state_int: State integer. int.
        -> Reward.
        """

        if state_int == self.n_states - 1:
            return 1
        return 0

    def average_reward(self, n_trajectories, trajectory_length, policy):
        """
        Calculate the average total reward obtained by following a given policy
        over n_paths paths.

        policy: Map from state integers to action integers.
        n_trajectories: Number of trajectories. int.
        trajectory_length: Length of an episode. int.
        -> Average reward, standard deviation.
        """

        trajectories = self.generate_trajectories(n_trajectories,
                                             trajectory_length, policy)
        rewards = [[r for _, _, r in trajectory] for trajectory in trajectories]
        rewards = np.array(rewards)

        # Add up all the rewards to find the total reward.
        total_reward = rewards.sum(axis=1)

        # Return the average reward and standard deviation.
        return total_reward.mean(), total_reward.std()

    def optimal_policy(self, state_int):
        """
        The optimal policy for this gridworld.

        state_int: What state we are in. int.
        -> Action int.
        """

        sx, sy = self.int_to_point(state_int)

        if sx < self.grid_size and sy < self.grid_size:
            return rn.randint(0, 2)
        if sx < self.grid_size-1:
            return 0
        if sy < self.grid_size-1:
            return 1
        raise ValueError("Unexpected state.")

    def optimal_policy_deterministic(self, state_int):
        """
        Deterministic version of the optimal policy for this gridworld.

        state_int: What state we are in. int.
        -> Action int.
        """

        sx, sy = self.int_to_point(state_int)
        if sx < sy:
            return 0
        return 1
    
    def my_generate_trajectories(self, policy,
                                    random_start=False):
        traj=[
        [68,69,68,67,66,56,56,56,56],
        [82,72,72,72,72,72,72,72,72],
        [74,64,63,63,63,63,63,63,63],
        [76,86,85,85,85,85,85,85,85],
        [1,2,12,12,12,12,12,12,12],
        [81,71,72,72,72,72,72,72,72],
        [35,45,55,56,56,56,56,56,56],
        [14,15,16,16,16,16,16,16,16],
        [17,16,16,16,16,16,16,16,16],
        [47,57,56,56,56,56,56,56,56],
        [11,12,12,12,12,12,12,12,12],
        [38,28,27,27,27,27,27,27,27],
        [98,88,87,97,87,86,85,85,85],
        [81,81,71,72,72,72,72,72,72],
        [10,10,11,21,21,22,22,22,22],
        [36,26,27,27,27,27,27,27,27],
        [97,87,97,96,86,85,85,85,85],
        [7,6,16,16,16,16,16,16,16],
        [22,22,22,22,22,22,22,22,22],
        [8,7,6,16,16,16,16,16,16],
        [72,72,72,72,72,72,72,72,72],
        [84,85,85,85,85,85,85,85,85],
        [42,43,53,63,63,63,63,63,63],
        [51,52,53,52,53,63,63,63,63],
        [75,75,85,85,85,85,85,85,85],
        [87,86,85,85,85,85,85,85,85],
        [3,2,12,12,12,12,12,12,12],
        [66,56,56,56,56,56,56,56,56],
        [71,72,72,72,72,72,72,72,72],
        [29,19,18,17,16,16,16,16,16],
        [11,12,12,12,12,12,12,12,12],
        [67,66,56,56,56,56,56,56,56],
        [39,29,19,18,17,16,16,16,16],
        [66,66,56,56,56,56,56,56,56],
        [34,34,35,45,55,56,56,56,56],
        [27,27,27,27,27,27,27,27,27],
        [30,40,41,42,32,22,22,22,22],
        [24,14,24,25,15,16,16,16,16],
        [96,97,87,86,85,85,85,85,85],
        [38,28,27,27,27,27,27,27,27],
        [86,85,85,85,85,85,85,85,85],
        [81,71,72,72,72,72,72,72,72],
        [60,61,62,63,63,63,63,63,63],
        [85,85,85,85,85,85,85,85,85],
        [90,80,70,60,61,62,72,72,72],
        [36,36,26,16,16,16,16,16,16],
        [32,32,22,22,22,22,22,22,22],
        [29,19,18,8,7,6,16,16,16],
        [9,8,7,6,16,16,16,16,16],
        [3,2,12,12,12,12,12,12,12],
        [47,48,58,57,56,56,56,56,56],
        [25,15,25,15,16,16,16,16,16],
        [90,90,80,70,60,70,60,61,62],
        [81,71,72,72,72,72,72,72,72],
        [67,67,66,56,56,56,56,56,56],
        [97,87,86,85,85,85,85,85,85],
        [19,18,17,16,16,16,16,16,16],
        [25,15,16,16,16,16,16,16,16],
        [45,46,56,56,56,56,56,56,56],
        [40,50,60,61,62,63,63,63,63],
        [52,53,63,63,63,63,63,63,63],
        [67,66,56,56,56,56,56,56,56],
        [81,71,72,72,72,72,72,72,72],
        [60,61,62,63,63,63,63,63,63],
        [21,21,21,21,21,21,21,21,21],
        [15,16,16,16,16,16,16,16,16],
        [43,53,63,63,63,63,63,63,63],
        [33,43,53,63,63,63,63,63,63],
        [57,56,56,56,56,56,56,56,56],
        [15,16,16,16,16,16,16,16,16],
        [48,58,57,56,56,56,56,56,56],
        [79,89,88,87,86,85,85,85,85],
        [14,15,16,16,16,16,16,16,16],
        [59,69,68,67,66,56,56,56,56],
        [72,72,72,72,72,72,72,72,72],
        [36,46,56,56,56,56,56,56,56],
        [47,47,57,56,56,56,56,56,56],
        [17,7,6,16,16,16,16,16,16],
        [54,53,63,63,63,63,63,63,63],
        [46,46,56,56,56,56,56,56,56],
        [58,57,56,56,56,56,56,56,56],
        [62,63,63,63,63,63,63,63,63],
        [9,9,8,7,6,16,16,16,16],
        [36,26,16,16,16,16,16,16,16],
        [90,90,80,70,60,61,62,63,63],
        [36,26,16,16,16,16,16,16,16],
        [10,11,12,12,12,12,12,12,12],
        [44,54,44,54,53,63,63,63,63],
        [39,39,49,59,58,68,67,66,56],
        [94,94,94,94,94,94,94,94,94],
        [20,21,21,21,21,21,21,21,21],
        [14,15,16,16,16,16,16,16,16],
        [19,18,17,16,16,16,16,16,16],
        [58,57,56,56,56,56,56,56,56],
        [36,26,16,16,16,16,16,16,16],
        [9,8,7,6,16,16,16,16,16],
        [98,98,97,96,86,85,85,85,85],
        [56,56,56,56,56,56,56,56,56],
        [40,50,60,61,62,63,63,63,63],
        [32,32,22,22,22,22,22,22,22],
        [72,72,72,72,72,72,72,72,72],
        [51,52,53,63,63,63,63,63,63],
        [90,80,70,60,61,62,63,63,63],
        [31,31,21,21,22,22,22,22,22],
        [88,88,87,97,96,86,85,85,85],
        [92,91,90,90,80,70,60,61,51],
        [96,95,94,94,94,94,94,94,94],
        [3,2,12,12,12,12,12,12,12],
        [9,8,8,7,6,16,16,16,16],
        [25,15,16,16,16,16,16,16,16],
        [77,76,86,85,85,85,85,85,85],
        [97,87,86,85,85,85,85,85,85],
        [52,42,52,53,63,63,63,63,63],
        [48,58,57,56,56,56,56,56,56],
        [42,52,53,63,63,63,63,63,63],
        [39,49,59,58,57,56,56,56,56],
        [84,85,85,85,85,85,85,85,85],
        [47,57,56,56,56,56,56,56,56],
        [88,87,86,85,85,85,85,85,85],
        [71,71,72,72,72,72,72,72,72],
        [66,56,56,56,56,56,56,56,56],
        [56,56,56,56,56,56,56,56,56]
        ]
        trajectories = []
        n_trajectories=len(traj)
        for i in range(n_trajectories):
            trajectory = []
            #for j in range(trajectory_length):
            for j in range(len(traj[i])):
                trajectory.append((traj[i][j], 0, 0))
            
            trajectories.append(trajectory)
        print(trajectories)
        return np.array(trajectories)
    
    
    def my_generate_trajectories_some_without_goal(self, policy,
                                    random_start=False):
        traj=[[0,1,2,3,8,13,12,17,22,23,24,-1,-1,-1,-1],
        [0,1,2,3,8,13,12,17,22,23,24,-1,-1,-1,-1],
        [0,1,2,3,8,13,12,17,22,23,24,-1,-1,-1,-1],
        [0,1,2,3,8,13,12,17,22,23,24,-1,-1,-1,-1],
        [0,1,2,3,4,9,14,13,12,17,16,21,22,23,24],
        [0,1,2,3,8,13,12,17,16,15,20,21,22,23,24],
        [0,1,2,3,8,13,12,17,22,23,24,-1,-1,-1,-1],
        [0,1,2,3,8,13,12,17,22,23,24,-1,-1,-1,-1],
        [0,1,2,3,8,13,12,17,22,23,24,-1,-1,-1,-1],
        [0,1,2,3,8,13,12,17,16,21,22,23,24,-1,-1],
        [0,1,2,3,8,13,12,17,16,15,20,21,22,23,24],
        [0,1,2,3,8,13,12,17,22,23,24,-1,-1,-1,-1],
        [0,1,2,3,4,9,14,13,12,17,22,23,24,-1,-1],
        [0,1,2,3,8,13,12,17,16,15,20,21,22,23,24],
        [0,1,2,3,8,13,12,17,22,23,24,-1,-1,-1,-1],
        [0,1,2,3,8,13,12,17,22,23,24,-1,-1,-1,-1],
        [0,1,2,3,8,13,12,17,22,23,24,-1,-1,-1,-1],
        [0,1,2,3,8,13,12,17,22,23,24,-1,-1,-1,-1],
        [0,1,2,3,8,13,12,17,22,23,24,-1,-1,-1,-1],
        [0,1,2,3,8,13,12,17,22,23,24,-1,-1,-1,-1],
        [0,1,2,3,8,13,12,17,22,23,24,-1,-1,-1,-1],
        [0,1,2,3,8,13,12,17,22,23,24,-1,-1,-1,-1],
        [0,1,2,3,8,13,12,17,22,23,24,-1,-1,-1,-1],
        [0,1,2,3,8,13,12,17,22,23,24,-1,-1,-1,-1],
        [0,1,2,3,8,13,12,17,22,23,24,-1,-1,-1,-1],
        [0,1,2,3,8,13,12,17,22,23,24,-1,-1,-1,-1],
        [0,1,2,3,8,13,12,17,22,23,24,-1,-1,-1,-1],
        [0,1,2,3,8,13,12,17,22,23,24,-1,-1,-1,-1],
        [0,1,2,3,8,13,12,17,22,23,24,-1,-1,-1,-1],
        [0,1,2,3,8,13,12,17,22,23,24,-1,-1,-1,-1],
        [0,1,2,3,8,13,12,17,22,23,24,-1,-1,-1,-1],
        [0,1,2,3,8,13,12,17,22,23,24,-1,-1,-1,-1],
        [0,1,2,3,8,13,12,17,22,23,24,-1,-1,-1,-1],
        [0,1,2,3,8,13,12,17,22,23,24,-1,-1,-1,-1],
        [0,1,2,3,8,13,12,17,22,23,24,-1,-1,-1,-1],
        [0,1,2,3,8,13,12,17,22,23,24,-1,-1,-1,-1],
        [0,1,2,3,8,13,12,17,22,23,24,-1,-1,-1,-1],
        [0,1,2,3,8,13,12,17,22,23,24,-1,-1,-1,-1],
        [0,1,2,3,8,13,12,17,22,23,24,-1,-1,-1,-1],
        
        [0,1,2,3,4,9,14,13,12,17,16,15,20,-1,-1],
        [0,1,2,3,4,9,14,13,12,17,16,15,20,-1,-1],
        [0,1,2,3,4,9,14,13,12,17,16,15,20,-1,-1],
        [0,1,2,3,4,9,14,13,12,17,16,15,20,-1,-1],
        [0,1,2,3,4,9,14,13,12,17,16,15,20,-1,-1],
        [0,1,2,3,4,9,14,13,12,17,16,15,20,-1,-1],
        [0,1,2,3,4,9,14,13,12,17,16,15,20,-1,-1],
        [0,1,2,3,4,9,14,13,12,17,16,15,20,-1,-1],
        [0,1,2,3,4,9,14,13,12,17,16,15,20,-1,-1],
        [0,1,2,3,4,9,14,13,12,17,16,15,20,-1,-1],
        [0,1,2,3,4,9,14,13,12,17,16,15,20,-1,-1],
        [0,1,2,3,4,9,14,13,12,17,16,15,20,-1,-1],
        [0,1,2,3,4,9,14,13,12,17,16,15,20,-1,-1],
        [0,1,2,3,4,9,14,13,12,17,16,15,20,-1,-1],
        [0,1,2,3,4,9,14,13,12,17,16,15,20,-1,-1],
        [0,1,2,3,4,9,14,13,12,17,16,15,20,-1,-1],
        [0,1,2,3,4,9,14,13,12,17,16,15,20,-1,-1],
        [0,1,2,3,4,9,14,13,12,17,16,15,20,-1,-1],
        [0,1,2,3,4,9,14,13,12,17,16,15,20,-1,-1],
        [0,1,2,3,4,9,14,13,12,17,16,15,20,-1,-1],
        [0,1,2,3,4,9,14,13,12,17,16,15,20,21,22],
        [0,1,2,3,4,9,14,13,12,17,16,15,20,21,22],
        [0,1,2,3,4,9,14,13,12,17,16,15,20,21,22],
        [0,1,2,3,4,9,14,13,12,17,16,15,20,21,22],
        [0,1,2,3,4,9,14,13,12,17,16,15,20,21,22],
        [0,1,2,3,4,9,14,13,12,17,16,15,20,21,22],
        [0,1,2,3,4,9,14,13,12,17,16,15,20,21,22],
        [0,1,2,3,4,9,14,13,12,17,16,15,20,21,22],
        [0,1,2,3,4,9,14,13,12,17,16,15,20,21,22],
        [0,1,2,3,4,9,14,13,12,17,16,15,20,21,22],
        [0,1,2,3,4,9,14,13,12,17,16,15,20,21,22],
        [0,1,2,3,4,9,14,13,12,17,16,15,20,21,22],
        [0,1,2,3,4,9,14,13,12,17,16,15,20,21,22],
        [0,1,2,3,4,9,14,13,12,17,16,15,20,21,22],
        [0,1,2,3,4,9,14,13,12,17,16,15,20,21,22],
        [0,1,2,3,4,9,14,13,12,17,16,15,20,21,22],
        [0,1,2,3,4,9,14,13,12,17,16,15,20,21,22],
        [0,1,2,3,4,9,14,13,12,17,16,15,20,21,22],
        [0,1,2,3,4,9,14,13,12,17,16,15,20,21,22],
        [0,1,2,3,4,9,14,13,12,17,16,15,20,21,22],
        
        [0,1,2,3,4,9,14,13,12,17,22,21,-1,-1,-1],
        [0,1,2,3,4,9,14,13,12,17,22,21,-1,-1,-1],
        [0,1,2,3,4,9,14,13,12,17,22,21,-1,-1,-1],
        [0,1,2,3,4,9,14,13,12,17,22,21,-1,-1,-1],
        [0,1,2,3,4,9,14,13,12,17,22,21,-1,-1,-1],
        [0,1,2,3,4,9,14,13,12,17,22,21,-1,-1,-1],
        [0,1,2,3,4,9,14,13,12,17,22,21,-1,-1,-1],
        [0,1,2,3,4,9,14,13,12,17,22,21,-1,-1,-1],
        [0,1,2,3,4,9,14,13,12,17,22,21,-1,-1,-1],
        [0,1,2,3,4,9,14,13,12,17,22,21,-1,-1,-1],
        [0,1,2,3,4,9,14,13,12,17,22,21,-1,-1,-1],
        [0,1,2,3,4,9,14,13,12,17,22,21,-1,-1,-1],
        [0,1,2,3,4,9,14,13,12,17,22,21,-1,-1,-1],
        [0,1,2,3,4,9,14,13,12,17,22,21,-1,-1,-1],
        [0,1,2,3,4,9,14,13,12,17,22,21,-1,-1,-1],
        [0,1,2,3,4,9,14,13,12,17,22,21,-1,-1,-1],
        [0,1,2,3,4,9,14,13,12,17,22,21,-1,-1,-1],
        [0,1,2,3,4,9,14,13,12,17,22,21,-1,-1,-1],
        [0,1,2,3,4,9,14,13,12,17,22,21,-1,-1,-1],
        [0,1,2,3,4,9,14,13,12,17,22,21,-1,-1,-1],
        [0,1,2,3,4,9,14,13,12,17,22,21,20,15,16],
        [0,1,2,3,4,9,14,13,12,17,22,21,20,15,16],
        [0,1,2,3,4,9,14,13,12,17,22,21,20,15,16],
        [0,1,2,3,4,9,14,13,12,17,22,21,20,15,16],
        [0,1,2,3,4,9,14,13,12,17,22,21,20,15,16],
        [0,1,2,3,4,9,14,13,12,17,22,21,20,15,16],
        [0,1,2,3,4,9,14,13,12,17,22,21,20,15,16],
        [0,1,2,3,4,9,14,13,12,17,22,21,20,15,16],
        [0,1,2,3,4,9,14,13,12,17,22,21,20,15,16],
        [0,1,2,3,4,9,14,13,12,17,22,21,20,15,16],
        [0,1,2,3,4,9,14,13,12,17,22,21,20,15,16],
        [0,1,2,3,4,9,14,13,12,17,22,21,20,15,16],
        [0,1,2,3,4,9,14,13,12,17,22,21,20,15,16],
        [0,1,2,3,4,9,14,13,12,17,22,21,20,15,16],
        [0,1,2,3,4,9,14,13,12,17,22,21,20,15,16],
        [0,1,2,3,4,9,14,13,12,17,22,21,20,15,16],
        [0,1,2,3,4,9,14,13,12,17,22,21,20,15,16],
        [0,1,2,3,4,9,14,13,12,17,22,21,20,15,16],
        [0,1,2,3,4,9,14,13,12,17,22,21,20,15,16],
        [0,1,2,3,4,9,14,13,12,17,22,21,20,15,16],
        ]
        trajectories = []
        n_trajectories=len(traj)
        for i in range(n_trajectories):
            trajectory = []
            #for j in range(trajectory_length):
            for j in range(len(traj[i])):
                trajectory.append((traj[i][j], 0, 0))
            
            trajectories.append(trajectory)
        
        return np.array(trajectories)
    
    def my_generate_trajectories_100(self, policy,random_start=False):
        traj=[
        [0,1,2,12,13,14,4,5,6,16,17,7,8,9,19,29,28,27,37,38,39,49,48,58,68,69,79,78,89,99],
        [0,1,11,12,2,3,13,14,15,5,6,16,26,27,28,38,48,58,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1],
        [0,10,20,30,40,50,60,70,71,72,73,83,93,94,95,96,97,98,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,10,11,12,13,14,15,16,26,27,28,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,17,27,37,47,48,58,68,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,14,15,16,26,27,28,38,48,58,68,78,88,98,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,17,27,37,38,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,16,26,27,37,47,48,58,68,78,88,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,16,17,27,28,38,48,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,16,17,27,28,38,48,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,8,9,19,29,39,49,59,69,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        
        
        [0,1,11,12,2,3,13,14,15,25,35,36,37,38,48,58,68,78,88,98,99,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,1,2,3,4,5,6,7,17,27,37,47,57,67,77,78,88,98,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [0,10,20,30,40,50,60,70,71,72,73,74,75,76,77,78,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        
        #[0,10,11,21,22,23,24,25,35,45,55,65,66,76,77,78,88,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        #[0,1,11,21,31,41,51,61,62,63,64,65,66,67,77,78,79,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        #[0,1,11,21,31,32,42,43,53,54,64,65,66,67,77,87,88,89,99,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        ]
        trajectories = []
        n_trajectories=len(traj)
        for i in range(n_trajectories):
            trajectory = []
            #for j in range(trajectory_length):
            for j in range(len(traj[i])):
                trajectory.append((traj[i][j], 0, 0))
            
            trajectories.append(trajectory)
        return np.array(trajectories)
    
    def feature_matrix_goalVsOther(self):
        features = [[0,1],[0,1],[0,1],[0,1],[0,1],
                    [0,1],[0,1],[0,1],[0,1],[0,1],
                    [0,1],[0,1],[0,1],[0,1],[0,1],
                    [0,1],[0,1],[0,1],[0,1],[0,1],
                    [0,1],[0,1],[0,1],[0,1],[1,0]]
        
        return np.array(features)
    
    def feature_matrix_goalVsOtherTwo(self):
        features = [[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],
                    [0,0,1],[0,0,1],[0,0,1],[0,1,0],[0,1,0],
                    [0,0,1],[0,0,1],[0,1,0],[0,1,0],[0,1,0],
                    [0,1,0],[0,1,0],[0,1,0],[0,0,1],[0,0,1],
                    [0,1,0],[0,1,0],[0,1,0],[0,1,0],[1,0,0]]
        
        return np.array(features)
    
    def feature_matrix_goalVsOtherThree(self):
        features = [[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,0,1,0],
                    [0,0,0,1],[0,0,0,1],[0,0,0,1],[0,1,0,0],[0,0,1,0],
                    [0,0,0,1],[0,0,0,1],[0,1,0,0],[0,1,0,0],[0,0,1,0],
                    [0,0,1,0],[0,0,1,0],[0,1,0,0],[0,0,0,1],[0,0,0,1],
                    [0,0,1,0],[0,0,1,0],[0,1,0,0],[0,1,0,0],[1,0,0,0],]
        
        return np.array(features)
    
    def feature_matrix_100(self):
        features = [
                [0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],
                [0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],
                [0,1,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],
                [0,1,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[0,1,0],[0,1,0],[0,1,0],
                [0,1,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[0,1,0],[0,1,0],[0,1,0],
                [0,1,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[0,1,0],[0,1,0],
                [0,1,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,0,0],[0,1,0],[0,1,0],
                [0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[1,0,0],[1,0,0],[0,1,0],[0,1,0],
                [0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],
                [0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,0,1]
                    ]
        return np.array(features)
    
    def feature_matrix_100_68(self):
        features=np.zeros(shape=(100,68))
        avoid_states=[21,22,23,24,25,31,32,33,34,35,36,41,42,43,44,45,46,51,52,53,54,55,56,57,61,62,63,64,65,66,67,76,77]
        avoid_states_feature=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                              0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
        print("avoid", len(avoid_states_feature))
        count=0
        for i in range(100):
            if(i in avoid_states):
                features[i]=avoid_states_feature
            else:
                feature=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                              0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                feature[count]=1
                features[i]=feature
                count+=1
                
        return np.array(features)