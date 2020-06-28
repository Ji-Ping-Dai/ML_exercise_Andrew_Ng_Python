# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 13:38:21 2020
Some functions to preform Anomaly Detection and Collaborative Filtering
@author: Ji-Ping Dai
"""
import numpy as np
import matplotlib.pyplot as plt 
plt.rc('text',usetex=True)
plt.rc('font',family='Times New Roman')

def estimateGaussian(X):
    '''Computes the probability density function of the multivariate gaussian distribution.'''
    mu = np.mean(X,axis=0).reshape(-1,1)
    sigma2 = (np.std(X,axis=0)**2).reshape(-1,1)
    return mu,sigma2

def multivariateGaussian(X,mu,sigma2):
    '''Computes the probability density function of the multivariate gaussian distribution.'''
    k = len(mu)
    sigma2 = np.diag(sigma2.ravel())
    P = np.diagonal((2*np.pi)**(-k/2)*np.linalg.det(sigma2)**(-0.5)*\
    np.exp(-0.5*(X-mu.ravel())@np.linalg.inv(sigma2)@(X-mu.ravel()).T))
    return P
    

def visualizeFit(X, mu, sigma2, ax):
    '''Visualize the dataset and its estimated distribution'''
    ax.plot(X[:,0],X[:,1],'bx')
    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('Throughput (mb/s)')
    x = np.linspace(4,24,100)
    y = np.linspace(4,24,100)
    px, py = np.meshgrid(x,y)
    P = multivariateGaussian(np.hstack([px.ravel().reshape(-1,1),py.ravel().reshape(-1,1)]),mu,sigma2).reshape(100,100)
    ax.contour(px,py,P,np.logspace(-10,-2,5),colors='k')
    return

def selectThreshold(yval,pval):
    '''Find the best threshold (epsilon) to use for selecting outliers'''
    bestEpsilon = 0
    bestF1 = 0
    F1 = 0
    epsilons = np.linspace(np.min(pval),np.max(pval),1000)
    for i in range(1,len(epsilons)-1):
        P = np.sum((pval<epsilons[i]) * (yval.ravel()==1))/np.sum(pval<epsilons[i])
        R = np.sum((pval<epsilons[i]) * (yval.ravel()==1))/np.sum(yval.ravel()==1)
        F1 = 2*P*R/(R+P)
        
        if F1>bestF1:
            bestF1 = F1
            bestEpsilon = epsilons[i]
    return bestEpsilon,bestF1

def cofiCostFunc(params, Y, R, num_users, num_movies, num_features, lam):
    ''' Collaborative filtering cost function'''
    X = params[:num_movies*num_features].reshape(num_movies,num_features)
    Theta = params[num_movies*num_features:].reshape(num_users, num_features)
    J = 0.5*np.sum((X@Theta.T-Y)*(X@Theta.T-Y)*R)+lam/2*(np.sum(np.diag(X.T@X))+np.sum(np.diag(Theta.T@Theta)))
    return J

def cofigrad(params, Y, R, num_users, num_movies, num_features, lam):
    X = params[:num_movies*num_features].reshape(num_movies,num_features)
    Theta = params[num_movies*num_features:].reshape(num_users, num_features)
    X_grad = ((X@Theta.T-Y)*R)@Theta + lam*X
    Theta_grad = ((X@Theta.T-Y)*R).T@X + lam*Theta
    grad = np.concatenate([X_grad.ravel(),Theta_grad.ravel()])
    return grad