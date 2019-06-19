"""
Run maximum entropy inverse reinforcement learning on the objectworld MDP.

Matthew Alger, 2015
matthew.alger@anu.edu.au
"""

import numpy as np
import matplotlib.pyplot as plt

import deep_maxent as deep_maxent
import gridworld as gridworld

def main(grid_size, discount, n_trajectories, epochs,learning_rate, structure):

    wind = 0.3
    trajectory_length = 8
    l1 = l2 = 0

    
    gw = gridworld.Gridworld(grid_size, wind, discount)
    
    ground_r = np.array([gw.reward(s) for s in range(gw.n_states)])
    
    
    trajectories = gw.my_generate_trajectories_some_without_goal(n_trajectories,trajectory_length,gw.optimal_policy)
    
    feature_matrix = gw.feature_matrix()
    
    r = deep_maxent.irl((feature_matrix.shape[1],) + structure, feature_matrix,
        gw.n_actions, discount, gw.transition_probability, trajectories, epochs,
        learning_rate, l1=l1, l2=l2)

    plt.subplot(1, 2, 1)
    plt.pcolor(ground_r.reshape((grid_size, grid_size)))
    plt.colorbar()
    plt.title("Groundtruth reward")
    plt.subplot(1, 2, 2)
    plt.pcolor(r.reshape((grid_size, grid_size)))
    plt.colorbar()
    plt.title("Recovered reward")
    plt.show()

if __name__ == '__main__':
    #main(10, 0.9, 15, 2, 20, 500, 0.01, (3, 3))
    main(5, 0.01, 30, 50, 0.01, (3, 3))
