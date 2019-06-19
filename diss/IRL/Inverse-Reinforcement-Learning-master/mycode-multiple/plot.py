#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 13:01:53 2019

@author: paras
"""

import matplotlib.pyplot as plt
def plot(ground_reward,reward,grid_size):
    plt.subplot(1, 2, 1)
    plt.pcolor(ground_reward.reshape((grid_size, grid_size)))
    plt.colorbar()
    plt.title("Groundtruth reward")
    plt.subplot(1, 2, 2)
    plt.pcolor(reward.reshape((grid_size, grid_size)))
    plt.colorbar()
    plt.title("Recovered reward")
    plt.show()