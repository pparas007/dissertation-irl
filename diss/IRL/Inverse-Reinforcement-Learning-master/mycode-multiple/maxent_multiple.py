from itertools import product

import numpy as np
import numpy.random as rn

import value_iteration
import math

def irl(feature_matrix, n_actions, discount, transition_probability,
        trajectories, epochs, learning_rate,trajectoryPerClusterWeights):
    n_states, d_states = feature_matrix.shape

    # Initialise weights.
    theta = rn.uniform(size=(d_states,))

    # Calculate the feature expectations \tilde{phi}. of trajectories
    feature_expectations = find_feature_expectations_weighted(feature_matrix,
                                                     trajectories,trajectoryPerClusterWeights)

    # Gradient descent on alpha.
    for i in range(epochs):
        r = feature_matrix.dot(theta)
        expected_svf = find_expected_svf(n_states, r, n_actions, discount,
                                         transition_probability, trajectories)
        grad = feature_expectations - feature_matrix.T.dot(expected_svf)

        theta += learning_rate * grad
        
    #return feature_matrix.dot(theta).reshape((n_states,))
    return theta

def find_svf(n_states, trajectories,trajectoryPerClusterWeight):

    svf = np.zeros(n_states)
    
    i=0
    for trajectory in trajectories:
        for state, _, _ in trajectory:
            svf[state] += (1*trajectoryPerClusterWeight[i])
        i+=1

    svf /= trajectories.shape[0]

    return svf

def find_feature_expectations(feature_matrix, trajectories):
    feature_expectations = np.zeros(feature_matrix.shape[1])

    for trajectory in trajectories:
        for state, _, _ in trajectory:
            feature_expectations += feature_matrix[state]
    feature_expectations /= trajectories.shape[0]
    return feature_expectations

def find_feature_expectations_weighted(feature_matrix, trajectories, trajectoryPerClusterWeights):
    feature_expectations = np.zeros(feature_matrix.shape[1])

    for i in range(np.size(trajectories,0)):
        for state, _, _ in trajectories[i]:
            feature_expectations += (trajectoryPerClusterWeights[i]*feature_matrix[state])
    feature_expectations /= trajectories.shape[0]
    return feature_expectations

def find_expected_svf(n_states, r, n_actions, discount,
                      transition_probability, trajectories):
    n_trajectories = trajectories.shape[0]
    trajectory_length = trajectories.shape[1]

    # policy = find_policy(n_states, r, n_actions, discount,
    #                                 transition_probability)
    policy = value_iteration.find_policy(n_states, n_actions,
                                         transition_probability, r, discount)

    start_state_count = np.zeros(n_states)
    for trajectory in trajectories:
        start_state_count[trajectory[0, 0]] += 1
    p_start_state = start_state_count/n_trajectories
    
    expected_svf = np.tile(p_start_state, (trajectory_length, 1)).T
    for t in range(1, trajectory_length):
        expected_svf[:, t] = 0
        for i, j, k in product(range(n_states), range(n_actions), range(n_states)):
            expected_svf[k, t] += (expected_svf[i, t-1] *
                                  policy[i, j] * # Stochastic policy
                                  transition_probability[i, j, k])

    return expected_svf.sum(axis=1)

def softmax(x1, x2):
    """
    Soft-maximum calculation, from algorithm 9.2 in Ziebart's PhD thesis.

    x1: float.
    x2: float.
    -> softmax(x1, x2)
    """

    max_x = max(x1, x2)
    min_x = min(x1, x2)
    return max_x + np.log(1 + np.exp(min_x - max_x))

def find_policy(n_states, r, n_actions, discount,
                           transition_probability):
    """
    Find a policy with linear value iteration. Based on the code accompanying
    the Levine et al. GPIRL paper and on Ziebart's PhD thesis (algorithm 9.1).

    n_states: Number of states N. int.
    r: Reward. NumPy array with shape (N,).
    n_actions: Number of actions A. int.
    discount: Discount factor of the MDP. float.
    transition_probability: NumPy array mapping (state_i, action, state_k) to
        the probability of transitioning from state_i to state_k under action.
        Shape (N, A, N).
    -> NumPy array of states and the probability of taking each action in that
        state, with shape (N, A).
    """

    # V = value_iteration.value(n_states, transition_probability, r, discount)

    # NumPy's dot really dislikes using inf, so I'm making everything finite
    # using nan_to_num.
    V = np.nan_to_num(np.ones((n_states, 1)) * float("-inf"))

    diff = np.ones((n_states,))
    while (diff > 1e-4).all():  # Iterate until convergence.
        new_V = r.copy()
        for j in range(n_actions):
            for i in range(n_states):
                new_V[i] = softmax(new_V[i], r[i] + discount*
                    np.sum(transition_probability[i, j, k] * V[k]
                           for k in range(n_states)))

        # # This seems to diverge, so we z-score it (engineering hack).
        new_V = (new_V - new_V.mean())/new_V.std()

        diff = abs(V - new_V)
        V = new_V

    # We really want Q, not V, so grab that using equation 9.2 from the thesis.
    Q = np.zeros((n_states, n_actions))
    for i in range(n_states):
        for j in range(n_actions):
            p = np.array([transition_probability[i, j, k]
                          for k in range(n_states)])
            Q[i, j] = p.dot(r + discount*V)

    # Softmax by row to interpret these values as probabilities.
    Q -= Q.max(axis=1).reshape((n_states, 1))  # For numerical stability.
    Q = np.exp(Q)/np.exp(Q).sum(axis=1).reshape((n_states, 1))
    return Q

def logLikelihoodOfTrajectory(trajectory,theta,feature_matrix,gw,discount):
     n_states, d_states = feature_matrix.shape
     r = feature_matrix.dot(theta)
     transition_probability=gw.transition_probability
     n_actions=gw.n_actions
     Q = value_iteration.find_policy(n_states, n_actions,
                                         transition_probability, r, discount)
     
     logLike = 0.0
     for i in range( (np.size(trajectory,0)-1) ):
         start_state=trajectory[i][0]
         next_state=trajectory[i+1][0]
         
         mostProbableAction=0
         mostProbableActionsProbability=gw.transition_probability[start_state][0][next_state]
         for action in range(n_actions):
             currentActionsProbability=gw.transition_probability[start_state][action][next_state]
             if(currentActionsProbability>mostProbableActionsProbability):
                 mostProbableActionsProbability=currentActionsProbability
                 mostProbableAction=action
                 
         actProb = Q[start_state][mostProbableAction]
         logLike += math.log(actProb)
        
     return logLike