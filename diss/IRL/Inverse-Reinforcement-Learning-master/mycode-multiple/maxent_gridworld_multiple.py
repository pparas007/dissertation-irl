#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 13:18:52 2019

@author: paras
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import maxent_multiple as maxent
import gridworld as gridworld
import math

def main(grid_size, discount, n_trajectories, epochs, learning_rate,wind):
    trajectory_length = 3*grid_size
    gw = gridworld.Gridworld(grid_size, wind, discount)
    trajectories = gw.my_generate_trajectories_multiple(n_trajectories,trajectory_length,gw.optimal_policy)
    feature_matrix = gw.feature_matrix()
    n_states, d_states = feature_matrix.shape
    
    theta=np.random.rand(2,25)
    clusterPriors=np.full((2,), 1.0/2.0)
    for i in range (15):
        trajectoryPerClusterWeights = computePerClusterMLIRLWeights(theta,clusterPriors,trajectories,feature_matrix,gw,discount)
        print("Iteration ",i)
        print("cluster priors", clusterPriors)
        print("trajectoryPerClusterWeights 0", trajectoryPerClusterWeights[0])
        print("trajectoryPerClusterWeights 1", trajectoryPerClusterWeights[1])
        for j in range (2):
            theta[j]=performIRL(epochs, learning_rate,wind,trajectoryPerClusterWeights[j],gw,trajectories,discount,grid_size,feature_matrix)    

def performIRL(epochs, learning_rate,wind,trajectoryPerClusterWeights,gw,trajectories,discount,grid_size,feature_matrix):
    n_states, d_states = feature_matrix.shape
    theta = maxent.irl(feature_matrix, gw.n_actions, discount,
        gw.transition_probability, trajectories, epochs, learning_rate,trajectoryPerClusterWeights)
    recovered_reward=feature_matrix.dot(theta).reshape((n_states,))
    
    scaler = StandardScaler()
    standardised_reward=scaler.fit_transform(recovered_reward.reshape(-1,1))
    
    plt.subplot(1, 2, 1)
    plt.pcolor(standardised_reward.reshape((grid_size, grid_size)))
    plt.colorbar()
    plt.title("Recovered reward")
    plt.show()
    
    return theta

def computePerClusterMLIRLWeights(theta,clusterPriors,trajectories,feature_matrix,gw,discount):
    k = 2
    n = 20
    newWeights = np.random.rand(k,n)
        
    for i in range (k):
        logPrior = math.log(clusterPriors[i])
        for j in range (n):
            trajectLogLikelihood = maxent.logLikelihoodOfTrajectory(trajectories[j],theta[i],feature_matrix,gw,discount)
            val= logPrior + trajectLogLikelihood
            newWeights[i][j] = val

    matrixSum = 0.0
    for j in range (n):
        columnDenom = computeClusterTrajectoryLoggedNormalization(j, newWeights)
        for i in range (k):
            logProb = newWeights[i][j] - columnDenom
            prob = math.exp(logProb)
            newWeights[i][j] = prob
            matrixSum += prob

    for i in range (k):
        clusterSum = 0.0
        for j in range (n):
            clusterSum += newWeights[i][j]
            
        nPrior = clusterSum / matrixSum
        clusterPriors[i] = nPrior

    return newWeights

def computeClusterTrajectoryLoggedNormalization(t, logWeightedLikelihoods):
    mx = float('-inf')
    k = (np.size(logWeightedLikelihoods,0))
    
    for i in range (k):
        mx = max(mx, logWeightedLikelihoods[i][t])

    sum = 0.0
    for i in range (k):
        v = logWeightedLikelihoods[i][t]
        shifted = v - mx
        exponentiated = math.exp(shifted)
        sum += exponentiated

    logSum = math.log(sum)
    finalSum = mx + logSum

    return finalSum
        
if __name__ == '__main__':
    main(5, 0.01, 20, 1000, 0.01,0.3)
