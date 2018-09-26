#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 21:14:41 2018

@author: me495
"""

import unittest
import numpy as np
import pandas as pd
import logistic_regression

class LogisticRegressionTest(unittest.TestCase):
    X = np.ones((4,3))
    Y = np.ones((3,1))
    
    '''
    X = [[1,0,1],
         [1,1,0],
         [1,2,1],
         [1,1,1]]
    Y = [[1],
         [0],
         [1],
         [1]]
    '''
    def tearDown(self):
        self.X[0][1] = 0
        self.X[1][2] = 0
        self.X[2][1] = 2
        self.Y[1][0] = 0
        