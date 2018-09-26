#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 20:41:19 2018

@author: me495
"""
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmod(x):
    return 1/(1+math.exp(-x))

def derivation(X, Y, theta):
    m, n = X.shape
    direction = np.zeros((n, 1))
    g = sigmod(np.dot(X, theta)) - Y
    for j in range(n):
        for i in range(m):
            direction[j][0] +=
        direction[j][0] /= m
    
def gradient_descent(data, ratio):
    pass
