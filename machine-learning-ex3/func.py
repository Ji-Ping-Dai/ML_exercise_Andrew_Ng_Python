# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 11:56:32 2020
Some functions to do Logistic Regression and Neural Network Learning
@author: Ji-Ping Dai
"""
import numpy as np
import matplotlib.pyplot as plt 
from scipy.optimize import minimize
plt.rc('text',usetex=True)
plt.rc('font',family='Times New Roman')


def sigmoid(z):
    '''sigmoid function, output between 0-1'''
    sigmoid = 1/(1+np.exp(-z))
    return sigmoid

def costfuncReg(theta,x,y,lam):
    '''Compute cost for Logistic Regression'''
    m = len(y)
    theta = theta.reshape(-1,1)
    J=np.zeros([len(theta),1])
    z = x@theta
    J = -1/m*(y.T@np.log(sigmoid(z))+(1-y).T@np.log(1-sigmoid(z)))+lam/2/m*theta.T[:,1:]@theta[1:,:]
    return J

def gridReg(theta,x,y,lam):
    '''Compute gradient for Logistic Regression'''
    m = len(y)
    theta = theta.reshape(-1,1)
    grid=np.zeros([len(theta),1])
    z = x@theta
    grid[0,:] = 1/m*(x.T@(sigmoid(z)-y))[0,:] 
    grid[1:,:] = 1/m*(x.T@(sigmoid(z)-y))[1:,:] + lam/m*theta[1:,:]
    return grid.flatten()

def onevsall(x, y, labels, lam):
    '''trains multiple logistic regression classifiers and returns all the classifiers in a matrix all_theta'''
    all_theta = np.zeros([labels,x.shape[1]])
    for i in range(labels):
        theta = np.zeros(x.shape[1])
        result = minimize(fun=costfuncReg, x0=theta, method='BFGS', jac=gridReg, args=(x,y==(i+1),lam))
        all_theta[i,:] = result.x
        print('final cost of label {0} is {1}'.format((i+1)%10,result.fun))
    return all_theta

def predictonevsall(all_theta,Xb):
    '''Predict the label for a trained one-vs-all classifier'''
    pred = sigmoid(Xb@all_theta.T)
    p = np.argmax(pred,axis=-1)+1
    return(p.reshape(-1,1))
    
def predictnn(Theta1, Theta2, Xb):
    '''Predict the label for a trained Neural Network classifier'''
    a1 = Xb
    a2 = np.zeros([Xb.shape[0],Theta1.shape[0]+1])
    a2[:,0] = 1
    a2[:,1:] = sigmoid(a1@Theta1.T)
    a3 = np.zeros([Xb.shape[0],Theta2.shape[0]])
    a3 = sigmoid(a2@Theta2.T)
    p = np.argmax(a3,axis=-1)+1
    return(p.reshape(-1,1))