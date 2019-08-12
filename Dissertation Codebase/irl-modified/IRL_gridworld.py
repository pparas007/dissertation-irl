import numpy as np
from sklearn.preprocessing import StandardScaler
import maxent as maxent
import objectworld as objectworld
import plot as plot
import deep_maxent as deep_maxent

def main(method):
    
    if(method=="linear"):
        print("linear method")
        theta = maxent.irl(feature_matrix, ow.n_actions, discount,
            ow.transition_probability, trajectories, epochs, learning_rate)
    elif(method=="deep"):
        print("deep method")
        l1 = l2 = 0
        theta = deep_maxent.irl((feature_matrix.shape[1],) + network_structure, feature_matrix,
            ow.n_actions, discount, ow.transition_probability, trajectories, epochs,
            learning_rate, l1=l1, l2=l2)
     
    recovered_reward=feature_matrix.dot(theta).reshape((n_states,))
    scaler = StandardScaler()
    standardised_reward=scaler.fit_transform(recovered_reward.reshape(-1,1))
    
    plot.plot(ground_r,standardised_reward, grid_size)


grid_size=10
#grid_size=10
n_objects=3
n_colours=3
discount=0.9
learning_rate=0.01
wind=0.3

ow = objectworld.Objectworld(grid_size, n_objects, n_colours, wind,discount)
ground_r = np.array([ow.reward(s) for s in range(ow.n_states)])
#feature_matrix = ow.feature_matrix_manhatten_distance()
feature_matrix = ow.feature_matrix_effect_range()
n_states, d_states = feature_matrix.shape
trajectories = ow.my_generate_trajectories2(ow.optimal_policy)

epochs=20
network_structure=(15,15)

if __name__ == '__main__':
    main(method="linear")
    #main(method="deep")
