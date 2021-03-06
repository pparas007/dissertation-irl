import numpy as np
from sklearn.preprocessing import StandardScaler
import maxent as maxent
import gridworld as gridworld
import plot as plot
import deep_maxent as deep_maxent

def main(method):
    if(method=="linear"):
        print("lenear method")
        recovered_reward = maxent.irl(feature_matrix, gw.n_actions, discount,
            gw.transition_probability, trajectories, epochs, learning_rate)
    elif(method=="deep"):
        print("deep method")
        l1 = l2 = 0
        recovered_reward = deep_maxent.irl((feature_matrix.shape[1],) + structure, feature_matrix,
            gw.n_actions, discount, gw.transition_probability, trajectories, epochs,
            learning_rate, l1=l1, l2=l2)
    scaler = StandardScaler()
    standardised_reward=scaler.fit_transform(recovered_reward.reshape(-1,1))
    
    plot.plot(ground_r,standardised_reward, grid_size)


grid_size=5
discount=0.01 
n_trajectories=25
epochs=700
learning_rate=0.01
wind=0.3
trajectory_length = 3*grid_size
gw = gridworld.Gridworld(grid_size, wind, discount)
ground_r = np.array([gw.reward(s) for s in range(gw.n_states)])

feature_matrix = gw.feature_matrix()
#feature_matrix = gw.feature_matrix_goalVsOther()
#feature_matrix = gw.feature_matrix_goalVsOtherTwo()
#feature_matrix = gw.feature_matrix_goalVsOtherThree()
feature_space=feature_matrix.shape[1]

#trajectories = gw.my_generate_trajectories(n_trajectories,trajectory_length,gw.optimal_policy)
trajectories = gw.my_generate_trajectories_some_without_goal(n_trajectories,trajectory_length,gw.optimal_policy)

n_states, d_states = feature_matrix.shape
no_of_iterations=20
structure=(3, 3)

if __name__ == '__main__':
    #main(method="linear")
    main(method="deep")
