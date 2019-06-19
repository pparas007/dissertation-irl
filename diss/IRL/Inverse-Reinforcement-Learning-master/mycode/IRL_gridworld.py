import numpy as np
from sklearn.preprocessing import StandardScaler
import maxent as maxent
import gridworld as gridworld
import plot as plot
import deep_maxent as deep_maxent

def main(method):
    
    if(method=="linear"):
        print("linear method")
        theta = maxent.irl(feature_matrix, gw.n_actions, discount,
            gw.transition_probability, trajectories, epochs, learning_rate)
    elif(method=="deep"):
        print("deep method")
        l1 = l2 = 0
        theta = deep_maxent.irl((feature_matrix.shape[1],) + network_structure, feature_matrix,
            gw.n_actions, discount, gw.transition_probability, trajectories, epochs,
            learning_rate, l1=l1, l2=l2)
    print(theta.shape)    
    recovered_reward=feature_matrix.dot(theta).reshape((n_states,))
    scaler = StandardScaler()
    standardised_reward=scaler.fit_transform(recovered_reward.reshape(-1,1))
    
    plot.plot(ground_r,standardised_reward, grid_size)


grid_size=5
grid_size=10

discount=0.9
learning_rate=0.01
wind=0.3

gw = gridworld.Gridworld(grid_size, wind, discount)
ground_r = np.array([gw.reward(s) for s in range(gw.n_states)])

#feature_matrix = gw.feature_matrix()
#feature_matrix = gw.feature_matrix_100()
feature_matrix = gw.feature_matrix_100_68()
#feature_matrix = gw.feature_matrix_goalVsOther()
#feature_matrix = gw.feature_matrix_goalVsOtherTwo()
#feature_matrix = gw.feature_matrix_goalVsOtherThree()
feature_space=feature_matrix.shape[1]
n_states, d_states = feature_matrix.shape

#trajectories = gw.my_generate_trajectories(gw.optimal_policy)
#trajectories = gw.my_generate_trajectories_some_without_goal(gw.optimal_policy)
trajectories = gw.my_generate_trajectories_100(gw.optimal_policy)

epochs=10
network_structure=(50,50)

if __name__ == '__main__':
    #main(method="linear")
    main(method="deep")
