# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 11:56:32 2020
Some functions to do Logistic Regression
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
    ax.plot(X[(y==1).flatten(),0], X[(y==1).flatten(),1], 'kx')
    ax.plot(X[(y==0).flatten(),0], X[(y==0).flatten(),1], 'yo')
    return

def sigmoid(z):
    '''sigmoid function, output between 0-1'''
    sigmoid = 1/(1+np.exp(-z))
    return sigmoid

def costfunc(theta,x,y):
    '''Compute cost for Logistic Regression'''
    m = len(y)
    theta = theta.reshape(-1,1)
    z = x@theta
    J = -1/m*(y.T@np.log(sigmoid(z))+(1-y).T@np.log(1-sigmoid(z)))
    return J

def costfuncReg(theta,x,y,lam):
    '''Compute cost for Logistic Regression'''
    m = len(y)
    theta = theta.reshape(-1,1)
    J=np.zeros([len(theta),1])
    z = x@theta
    J = -1/m*(y.T@np.log(sigmoid(z))+(1-y).T@np.log(1-sigmoid(z)))+lam/2/m*theta.T[:,1:]@theta[1:,:]
    return J

def grid(theta,x,y):
    '''Compute gradient for Logistic Regression'''
    m = len(y)
    theta = theta.reshape(-1,1)
    z = x@theta
    grid = 1/m*(x.T@(sigmoid(z)-y))
    return grid.flatten()

def gridReg(theta,x,y,lam):
    '''Compute gradient for Logistic Regression'''
    m = len(y)
    theta = theta.reshape(-1,1)
    grid=np.zeros([len(theta),1])
    z = x@theta
    grid[0,:] = 1/m*(x.T@(sigmoid(z)-y))[0,:] 
    grid[1:,:] = 1/m*(x.T@(sigmoid(z)-y))[1:,:] + lam/m*theta[1:,:]
    return grid.flatten()

def gradientDescent(x,y,theta,iterations,alpha):
    '''Performs gradient descent to learn theta'''
    m = len(y)
    cost = np.zeros(iterations)
    for i in range(iterations):
        z = x@theta
        theta = theta-1/m*alpha*x.T@(sigmoid(z)-y)
        cost[i] = costfunc(theta,x,y)
    return theta, cost

def plotboundary(X,y,theta,ax):
    '''Plots the Decision Boundary'''
    plotdata(X,y,ax)
    if len(theta)<=3:
        px=np.linspace(np.min(X[:,0])-2,np.max(X[:,0])+2,100)
        py=(-theta[0]-theta[1]*px)/theta[2]
        ax.plot(px,py,ls='--',color='b')
    else:
        px=np.linspace(-1,1.5,100)
        py=np.linspace(-1,1.5,100)
        cost=np.zeros([100,100])
        for i in range(len(px)):
            for j in range(len(py)):
                cost[i,j]=mapX(px[i],py[j])@theta
        cost=cost.T
        PX,PY=np.meshgrid(px,py)
        ax.contour(PX,PY,cost,[0],colors='k')
    return

def predict(x,theta):
    '''Predict whether the label is 0 or 1 using learned logistic '''
    p = sigmoid(x@theta.reshape(-1,1))
    p[p>=0.5]=1
    p[p<0.5]=0
    return p

def mapX(x1,x2):
    '''Feature mapping function to polynomial features'''
    order = 6
    X = np.ones([x1.size,1])
    order=6
    for i in range(1,order+1):
        for j in range(i+1):
            newcol = (x1**(i-j)*x2**(j)).reshape(-1,1)
            X = np.concatenate([X,newcol],axis=1)
    return X
    
    
    
    
    
    