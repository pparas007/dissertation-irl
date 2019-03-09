"""
Run maximum entropy inverse reinforcement learning on the gridworld MDP.

Matthew Alger, 2015
matthew.alger@anu.edu.au
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import irl.maxent as maxent
import irl.mdp.gridworld as gridworld
import math
def main(grid_size, discount, n_trajectories, epochs, learning_rate):
    """
    Run maximum entropy inverse reinforcement learning on the gridworld MDP.

    Plots the reward function.


    grid_size: Grid size. int.
    discount: MDP discount factor. float.
    n_trajectories: Number of sampled trajectories. int.
    epochs: Gradient descent iterations. int.
    learning_rate: Gradient descent learning rate. float.
    """

    wind = 0.3
    trajectory_length = 3*grid_size

    gw = gridworld.Gridworld(grid_size, wind, discount)
    
    #trajectories = gw.generate_trajectories(n_trajectories,trajectory_length,gw.optimal_policy)
    trajectories = gw.my_generate_trajectories(n_trajectories,trajectory_length,gw.optimal_policy)
    
    feature_matrix = gw.feature_matrix()
    #feature_matrix = gw.feature_matrix_goalVsOther()
    #feature_matrix = gw.feature_matrix_goalVsOtherTwo()
    #feature_matrix = gw.feature_matrix_goalVsOtherThree()
    
    #ground truth given by us as we know which states are good vs bad
    ground_r = np.array([gw.reward(s) for s in range(gw.n_states)])
    
    #reard recovered using IRL algorithm
    recovered_reward = maxent.irl(feature_matrix, gw.n_actions, discount,
        gw.transition_probability, trajectories, epochs, learning_rate)
    
    #let's standardiese it    
    scaler = StandardScaler()
    standardised_reward=scaler.fit_transform(recovered_reward.reshape(-1,1))
    #print(recovered_reward)
    #print(standardised_reward)    
    
    plt.subplot(1, 2, 1)
    plt.pcolor(ground_r.reshape((grid_size, grid_size)))
    plt.colorbar()
    plt.title("Groundtruth reward")
    plt.subplot(1, 2, 2)
    plt.pcolor(standardised_reward.reshape((grid_size, grid_size)))
    plt.colorbar()
    plt.title("Recovered reward")
    plt.show()

if __name__ == '__main__':
    main(5, 0.01, 30, 2000, 0.01)
