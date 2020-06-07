# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 11:56:32 2020
Some functions to do linear regression
@author: Ji-Ping Dai
"""
import numpy as np
import matplotlib.pyplot as plt 
plt.rc('text',usetex=True)
plt.rc('font',family='Times New Roman')

def plotdata(X,y,ax):
    '''PLOTDATA Plots the data points x and y into a new figure'''
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax.set_ylabel('Profit in \$10,000',size=20)
    ax.set_xlabel('Population of City in 10,000s',size=20)
    ax.plot(X, y, 'rx',label='Training data')
    return

def costfunc(x,y,theta):
    '''Compute cost for linear regression'''
    m = len(y)
    J = 1/2/m*np.sum((x@theta-y)**2)
    return J

def gradientDescent(x,y,theta,iterations,alpha):
    '''Performs gradient descent to learn theta'''
    m = len(y)
    cost = np.zeros(iterations)
    for i in range(iterations):
        theta = theta-1/m*alpha*x.T@(x@theta-y)
        cost[i] = costfunc(x,y,theta)
    return theta, cost

def featureNormalize(X):
    '''Normalizes the features in X '''
    mu = np.mean(X,axis=0)
    sigma = np.std(X,axis=0)
    X = (X-mu)/sigma
    return X, mu, sigma