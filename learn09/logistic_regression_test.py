#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 21:14:41 2018

@author: me495
"""

import unittest
import numpy as np
import pandas as pd
import logistic_regression as lr

class LogisticRegressionTest(unittest.TestCase):
    # 输入数据
    X = np.array([[1,0,1],
                  [1,1,0],
                  [1,2,1],
                  [1,1,1]])
    Y = np.array([[1],
                  [0],
                  [1],
                  [1]])
    theta = np.array([[1],
                      [0],
                      [1]])
    
    # Y1 = sigmod(X*theta) 手算出来的值，用来与正确结果作比较
    Y1 = np.array([[0.88079708],
                   [0.73105858],
                   [0.88079708],
                   [0.88079708]])
    # 手算出来的值，用来与正确结果作比较
    direction = np.array([[ 0.09336245],
                          [ 0.09336245],
                          [-0.08940219]])
    def test_sigmod(self):
        Y1 = lr.sigmod(np.dot(self.X, self.theta))
        self.assertEqual(np.sum(np.abs(Y1-self.Y1)>1e-8), 0)
        
    def test_derivation(self):
        direction = lr.calc_gradient(self.X, self.Y,self.theta)
        self.assertAlmostEqual(np.sum(np.abs(direction-self.direction)>1e-8), 0)
        
    def test_calc_accuracy(self):
        accuracy = lr.calc_accuracy(self.X, self.Y, self.theta)
        self.assertTrue(np.abs(accuracy-0.75)<1e-8)
        
if __name__ == '__main__':
    unittest.main()
        