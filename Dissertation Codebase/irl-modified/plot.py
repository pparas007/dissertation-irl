#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 13:01:53 2019

@author: paras
"""
import matplotlib.pyplot as plt
def plot(ground_reward,reward,grid_size):
    ground_reward=getGriundReward(ground_reward)
    plt.subplot(1, 2, 1)
    plt.pcolor(ground_reward.reshape((grid_size, grid_size)), linewidths=4)
    plt.colorbar()
    plt.title("Groundtruth reward")
    plt.subplot(1, 2, 2)
    plt.pcolor(reward.reshape((grid_size, grid_size)), linewidths=4)
    plt.colorbar()
    plt.title("Recovered reward")
    plt.tight_layout()
    plt.show()
    
def getGriundReward(ground_reward):
    non_reward_states=[12,21,22,16,27,56,63,72,85,94]
    reward_states=[0,1,2,10,11,13,20,31,25,26,34,35,36,37,38,45,46,47,73,74,81,82,83,84,92,93]
    for i in range(0,len(ground_reward)):
        ground_reward[i]=0
        if(i in reward_states):
            ground_reward[i]=1
        elif(i in non_reward_states):
            ground_reward[i]=-1
    print(ground_reward)
    return ground_reward