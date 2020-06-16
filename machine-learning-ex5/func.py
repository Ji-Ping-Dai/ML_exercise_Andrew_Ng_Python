# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 11:56:32 2020
Some functions to preform Regularized Linear Regression and Bias-Variance EXERSIZE 5
@author: Ji-Ping Dai
"""
import numpy as np
import matplotlib.pyplot as plt 
from scipy.optimize import minimize
plt.rc('text',usetex=True)
plt.rc('font',family='Times New Roman')

def costfunc(theta,x,y,lam):
    '''Compute cost for linear regression'''
    theta = theta.reshape(-1,1)
    m = len(y)
    J = 1/2/m*(x@theta-y).T@(x@theta-y) + lam/2/m*theta[1:].T@theta[1:]
    return J

def gradient(theta,x,y,lam):
    '''gradient cost for linear regression'''
    theta = theta.reshape(-1,1)
    m = len(y)
    grad = 1/m*x.T@(x@theta-y)
    grad[1:] = grad[1:]+lam/m*theta[1:] 
    return grad.reshape(1,-1)[0]

def trainLR(x,y,lam):
    '''Trains linear regression'''
    result = minimize(fun=costfunc,x0=np.zeros(x.shape[1]), method='BFGS', jac=gradient, args=(x,y,lam))
    return result.x

def learningCurve(x,y,xval,yval,lam):
    '''Generates the train and cross validation set errors needed to plot a learning curve'''
    m = x.shape[0]
    error_train = np.zeros(m)
    error_val = np.zeros(m)
    for i in range(m):
        theta = trainLR(x[:i+1],y[:i+1],lam)
        error_train[i] = costfunc(theta,x[:i+1],y[:i+1],0) #lam must be zero.
        error_val[i] = costfunc(theta,xval,yval,0)
    return error_train,error_val

def polyFeatures(x,p):
    '''POLYFEATURES Maps X (1D vector) into the p-th power'''
    x_poly=np.zeros([len(x),p])
    for i in range(p):
        x_poly[:,i] = x[:,0]**(i+1)
    return x_poly