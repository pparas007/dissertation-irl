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
import maxent as maxent
import gridworld as gridworld
import math

def main(grid_size, discount, n_trajectories, epochs, learning_rate,wind):
    print('here')
    t=newWeights = np.random.rand(1,2)
    print(t)
    fun(t)
    print(t)
def fun(t):
    t[0][0]=22
    t[0][1]=22      
if __name__ == '__main__':
    main(5, 0.01, 20, 200, 0.01,0.3)
