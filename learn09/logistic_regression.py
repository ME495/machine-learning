#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 20:41:19 2018

@author: me495
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BATCH = 0
STOCHASTIC = 1
SMALL_BATCH = 2

STOP_ITER = 0
STOP_COST = 1
STOP_GRAD = 2
x = np.array([])
y = np.array([])

def sigmod(x):
    return 1/(1+np.exp(-x))

def calc_cost(data, theta):
    cols = data.shape[1]
    X, Y = data[:, 0:cols-1], data[:, cols-1:]
    m = X.shape[0]
    left = (-Y)*np.log(sigmod(np.dot(X, theta)))
    right = (1-Y)*np.log(1-sigmod(np.dot(X, theta)))
    return np.sum(left-right)/m

def calc_gradient(X, Y, theta):
    m, n = X.shape
    grad = np.zeros((n, 1))
    g = sigmod(np.dot(X, theta)) - Y
    for j in range(n):
        for i in range(m):
            grad[j][0] += g[i]*X[i][j]
        grad[j][0] /= m
    return grad

def calc_mean_grad(data, theta):
    cols = data.shape[1]
    X, Y = data[:, 0:cols-1], data[:, cols-1:]
    return np.mean(np.abs(calc_gradient(X, Y, theta)))

def calc_accuracy(data, theta):
    cols = data.shape[1]
    X, Y = data[:, 0:cols-1], data[:, cols-1:]
    Y1 = np.dot(X, theta)
    Y1[Y1>0.5] = 1
    Y1[Y1<1] = 0
    m = Y1.shape[0]
    return np.sum(Y==Y1)/m
    
def is_stop(data, theta, stop_type, rest, cost, grad):
    if stop_type==STOP_ITER:
        return rest==0
    elif stop_type==STOP_COST:
        return calc_cost(data, theta)<cost
    elif stop_type==STOP_GRAD:
        return calc_mean_grad(data, theta)<grad
    
'''
data: 输入数据，数据的最后一列为Y
ratio: 学习率
descent_type: 梯度下降方式 
    BATCH: 批量梯度下降 
    STOCHASTIC: 随机梯度下降 
    SMALL_BATCH: 小批量梯度下降
stop_type: 停止方式
    STOP_ITER: 根据迭代次数停止
    STOP_COST: 根据正确率停止
    STOP_GRAD: 根据梯度停止
num: 每次梯度下降所使用的数据数量，可选，当descent_type==SMALL_BATCH时起作用
count: 迭代次数，当stop_type==STOP_ITER时起作用
cost: 要求达到的最小损失，当stop_type==STOP_COST时起作用
grad: 要求达到的最小梯度，当stop_type==STOP_GRAD时起作用
'''
def descent(data, ratio, descent_type, stop_type, num=None, count=None, cost=None, grad=None):
    global x, y
    m, cols = data.shape #m表示数据总数量，cols表示数据有多少列
    theta = np.zeros((cols-1, 1))
    begin, step = 0, 0
    if count==None:
        count = 0
    while True:
        step += 1
        if descent_type==BATCH:
            np.random.shuffle(data)
            X, Y = data[:, 0:cols-1], data[:, cols-1:]
            # print(ratio*derivation(X, Y, theta))
            theta -= ratio*calc_gradient(X, Y, theta)
        elif descent_type==STOCHASTIC:
            row_id = np.random.randint(0, m, 1)
            X, Y = data[row_id:row_id+1, 0:cols-1], data[row_id:row_id+1, cols-1:]
            theta -= ratio*calc_gradient(X, Y, theta)
        elif descent_type==SMALL_BATCH:
            if begin+num>m:
                np.random.shuffle(data)
                begin = 0
            X, Y = data[begin:begin+num, 0:cols-1], data[begin:begin+num, cols-1:]
            theta -= ratio*calc_gradient(X, Y, theta)
        x = np.append(x, step)
        y = np.append(y, calc_cost(data, theta))
        # print(calc_mean_grad(X, Y, theta))
        # print(calc_accuracy(X, Y, theta)) 
        # print(theta)
        # print("--------")
        
        if is_stop(data, theta, stop_type, count-step, cost, grad):
            # print(calc_accuracy(X, Y, theta))
            print(calc_accuracy(data, theta))
            return theta

def f(x, theta):
    return -(theta[0]+x*theta[1])/theta[2]
if __name__=='__main__':
    frame = pd.read_csv('LogiReg_data.txt', names=['exam1', 'exam2', 'accept'])
    data = np.c_[np.ones((frame.shape[0],1)),frame.values]
    theta = descent(data, 0.0003, SMALL_BATCH, STOP_ITER, num=20, count=60000)
    print(theta)
    plt.plot(x, y)
    plt.show()