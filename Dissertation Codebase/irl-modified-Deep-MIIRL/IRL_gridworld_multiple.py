#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 13:18:52 2019

@author: paras
"""

import numpy as np
import math

from sklearn.preprocessing import StandardScaler
import maxent_multiple as maxent
#import gridworld as gridworld
import objectworld as objectworld
import plot as plot
import deep_maxent_multiple as deep_maxent
import random
 
def miirl(method):
    theta=np.random.rand(no_clusters,feature_space)
    clusterPriors=np.full((no_clusters,), 1.0/no_clusters)
    for i in range (no_of_iterations):
        trajectoryPerClusterWeights = computePerClusterMLIRLWeights(theta,clusterPriors)
        #trajectoryPerClusterWeights=changeTrajectoryPerClusterWeights(trajectoryPerClusterWeights)
        
        print("Iteration ",i)
        print("cluster priors", clusterPriors)
        print("trajectoryPerClusterWeights 0", trajectoryPerClusterWeights[0])
        print("trajectoryPerClusterWeights 1", trajectoryPerClusterWeights[1])
        for j in range (no_clusters):
            theta[j]=performIRL(method,trajectoryPerClusterWeights[j])    

def performIRL(method,trajectoryPerClusterWeight):
    #theta = maxent.irl(feature_matrix, gw.n_actions, discount,
        #gw.transition_probability, trajectories, epochs, learning_rate,trajectoryPerClusterWeight)
    
    if(method=="linear"):
        print("linear method")
        theta = maxent.irl(feature_matrix, ow.n_actions, discount,
            ow.transition_probability, trajectories, epochs, learning_rate,trajectoryPerClusterWeight)
    elif(method=="deep"):
        print("deep method")
        l1 = l2 = 0
        theta = deep_maxent.irl((feature_matrix.shape[1],) + network_structure, feature_matrix,
            ow.n_actions, discount, ow.transition_probability, trajectories, epochs,
            learning_rate,trajectoryPerClusterWeight, l1=l1, l2=l2)
    
    recovered_reward=feature_matrix.dot(theta).reshape((n_states,))
    
    scaler = StandardScaler()
    standardised_reward=scaler.fit_transform(recovered_reward.reshape(-1,1))
    
    plot.plot(ground_r,standardised_reward, grid_size)
    
    return theta

def computePerClusterMLIRLWeights(theta,clusterPriors):
    newWeights = np.random.rand(no_clusters,n_trajectories)
        
    for i in range (no_clusters):
        logPrior = math.log(clusterPriors[i])
        for j in range (n_trajectories):
            trajectLogLikelihood = maxent.logLikelihoodOfTrajectory(trajectories[j],theta[i],feature_matrix,ow,discount)
            val= logPrior + trajectLogLikelihood
            newWeights[i][j] = val

    matrixSum = 0.0
    for j in range (n_trajectories):
        columnDenom = computeClusterTrajectoryLoggedNormalization(j, newWeights)
        for i in range (no_clusters):
            logProb = newWeights[i][j] - columnDenom
            prob = math.exp(logProb)
            newWeights[i][j] = prob
            matrixSum += prob

    for i in range (no_clusters):
        clusterSum = 0.0
        for j in range (n_trajectories):
            clusterSum += newWeights[i][j]
            
        nPrior = clusterSum / matrixSum
        clusterPriors[i] = nPrior

    return newWeights

def computeClusterTrajectoryLoggedNormalization(t, logWeightedLikelihoods):
    mx = float('-inf')
    
    for i in range (no_clusters):
        mx = max(mx, logWeightedLikelihoods[i][t])

    sum = 0.0
    for i in range (no_clusters):
        v = logWeightedLikelihoods[i][t]
        shifted = v - mx
        exponentiated = math.exp(shifted)
        sum += exponentiated

    logSum = math.log(sum)
    finalSum = mx + logSum

    return finalSum

def changeTrajectoryPerClusterWeights(trajectoryPerClusterWeights):
    for i in range(len(trajectoryPerClusterWeights[0])):
        if(i<253):
            trajectoryPerClusterWeights[0][i]=random.uniform(0, 0.3)
        else:
            trajectoryPerClusterWeights[0][i]=random.uniform(0.7,1)
    for i in range(len(trajectoryPerClusterWeights[1])):
        if(i<253):
            trajectoryPerClusterWeights[1][i]=random.uniform(0.7,1)
        else:
            trajectoryPerClusterWeights[1][i]=random.uniform(0, 0.3)
    return trajectoryPerClusterWeights

no_clusters=2
grid_size=10
discount=0.9
learning_rate=0.01
wind=0.3
n_objects=3
n_colours=3

trajectory_length = 9
ow = objectworld.Objectworld(grid_size, n_objects, n_colours, wind,discount)
ground_r = np.array([ow.reward(s) for s in range(ow.n_states)])
#feature_matrix = ow.feature_matrix_manhatten_distance()
feature_matrix = ow.feature_matrix_effect_range()

feature_space=feature_matrix.shape[1]

n_states, d_states = feature_matrix.shape

trajectories = ow.my_generate_trajectories2(ow.optimal_policy)

n_trajectories=trajectories.shape[0]
no_of_iterations=5
epochs=5
network_structure=(10 , 10)

if __name__ == '__main__':
    miirl(method="linear")
    #miirl(method="deep")
