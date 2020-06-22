# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 21:32:23 2020
Some functions to preform Principle Component Analysis and K-Means Clustering
@author: Ji-Ping Dai
"""
import numpy as np
import matplotlib.pyplot as plt 
plt.rc('text',usetex=True)
plt.rc('font',family='Times New Roman')

def findClosestCentroids(X, C):
    '''computes the centroid memberships for every example'''
    m = X.shape[0]
    idx = np.zeros(m,dtype = 'int64')
    for i in range(m):
        idx[i] = np.argmin(np.diag((C-X[i,:])@(C-X[i,:]).T))
    return idx

def computeCentroids(X, idx, K):
    '''returns the new centroids by computing the means of the data points assigned to each centroid.'''
    C = np.zeros([K,X.shape[1]])
    for i in range(K):
        C[i,:] = np.mean(X[idx == i,:],axis=0)
    return C

def runkMeans(X, IC, iters, plot_progress=False):
    '''runs the K-Means algorithm on data matrix X, where each row of X is a single example'''
    m = X.shape[0]
    K = IC.shape[0]
    C = IC
    PC = IC
    idx = np.zeros(m)
    
    for i in range(iters):
        print('K-Means iteration {0}/{1}...'.format(i+1, iters))
        idx = findClosestCentroids(X, C)
        PC = C
        C = computeCentroids(X, idx, K)
        if plot_progress:
            color = ['b','r','g']
            for j in range(K):
                #print(PC[j],C[j])
                plt.plot([PC[j,0], C[j,0]],[PC[j,1], C[j,1]],color = 'k', ls = '--',)
                plt.scatter([PC[j,0], C[j,0]],[PC[j,1], C[j,1]],marker = '<', c=color[j], s=50)
    if plot_progress:
        plt.scatter(X[idx==0,0],X[idx==0,1],color="w",linewidths=1,s=20,edgecolors='b',alpha=0.3)
        plt.scatter(X[idx==1,0],X[idx==1,1],color="w",linewidths=1,s=20,edgecolors='r',alpha=0.3)
        plt.scatter(X[idx==2,0],X[idx==2,1],color="w",linewidths=1,s=20,edgecolors='g',alpha=0.3)
        
    return C

def PCA(X):
    '''Run principal component analysis on the dataset X'''
    S, U = np.linalg.eig(X.T@X/X.shape[0])
    return U,S

def projectData(X,U,K):
    '''Computes the reduced data representation when projecting only on to the top k eigenvectors'''
    Z = X@U[:,:K]
    return Z
    
def recoverData(Z,U,K):
    '''Recovers an approximation of the original data when using the projected data'''
    X = Z@U[:,:K].T
    return X
    
    
    
    
    
    
    
    
    
    
    
    
    