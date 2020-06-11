# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 11:56:32 2020
Some functions to trian Neural Network EXERSIZE 4
@author: Ji-Ping Dai
"""
import numpy as np
import matplotlib.pyplot as plt 
from scipy.optimize import minimize
plt.rc('text',usetex=True)
plt.rc('font',family='Times New Roman')


def randiniweight(m,n):
    ''' Randomly initialize the weights of a layer with L_in incoming connections and L_out outgoing connections'''
    sigma = np.sqrt(6/(m+n+1))
    W =np.random.rand(n,m+1)*2*sigma-sigma
    return W        

def sigmoid(z):
    '''sigmoid function, output between 0-1'''
    sigmoid = 1/(1+np.exp(-z))
    return sigmoid

def sigmoidgrad(z):
    '''returns the gradient of the sigmoid function'''
    g = sigmoid(z) * (1-sigmoid(z))
    return g

def nncostfunc(nnTheta, X, y, lam, n1, n2, n3):
    '''Compute cost for Neural Network'''
    Theta1 = nnTheta[:n2*(n1+1)].reshape(n2,(n1+1))
    Theta2 = nnTheta[n2*(n1+1):].reshape(n3,(n2+1))
    m = X.shape[0]
    J = 0
    a1 = np.concatenate([np.ones([m, 1]), X], axis=1)
    a2 = np.concatenate([np.ones([m, 1]), sigmoid(a1@Theta1.T)], axis=1)
    ht = sigmoid(a2@Theta2.T)
    
    yt = np.zeros([m, n3])
    for i in range(m):
        yt[i, y[i]-1] = 1
    
    for i in range(n3):
        J = J-1/m*(yt[:,i].T @ np.log(ht[:,i]) + (1-yt[:,i]).T @ np.log(1-ht[:,i]))
    J = J+1/2/m*lam*(np.sum((Theta1**2)[:,1:]) + np.sum((Theta2**2)[:,1:]))
    return J

def nngradient(nnTheta, X, y, lam, n1, n2, n3):
    '''Implement the backpropagation algorithm to compute the gradients'''
    Theta1 = nnTheta[:n2*(n1+1)].reshape(n2,(n1+1))
    Theta2 = nnTheta[n2*(n1+1):].reshape(n3,(n2+1))
    m = X.shape[0]
    a1 = np.concatenate([np.ones([m, 1]), X], axis=1)
    a2 = np.concatenate([np.ones([m, 1]), sigmoid(a1@Theta1.T)], axis=1)
    ht = sigmoid(a2@Theta2.T)
    yt = np.zeros([m, n3])
    for i in range(m):
        yt[i, y[i]-1] = 1
    
    delta3 = ht-yt
    delta2 = (delta3 @ Theta2)[:,1:]*sigmoidgrad(a1@Theta1.T)
    # do not calculate theta1
    Theta2_grad = 1/m*delta3.T@a2
    Theta1_grad = 1/m*delta2.T@a1
    Theta2_grad[:,1:] = Theta2_grad[:,1:]+lam/m*Theta2[:,1:]
    Theta1_grad[:,1:] = Theta1_grad[:,1:]+lam/m*Theta1[:,1:]
    return np.concatenate([Theta1_grad.ravel(),Theta2_grad.ravel()])

def checkNNGradients(lam):
    '''Creates a small neural network to check the backpropagation gradients'''
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5
    Theta1 = randiniweight(input_layer_size, hidden_layer_size)
    Theta2 = randiniweight(hidden_layer_size, num_labels)
    X = np.random.rand(m, input_layer_size)
    y = np.random.randint(0,3,size=(m,1))
    nnTheta = np.concatenate([Theta1.ravel(),Theta2.ravel()])
    nngrad = nngradient(nnTheta, X, y, lam, input_layer_size, hidden_layer_size, num_labels)
    numgrad = np.zeros_like(nnTheta)
    perturb = np.zeros_like(nnTheta)
    e = 1e-4
    for i in range(len(numgrad)):
        perturb[i] = e
        numgrad[i] = (nncostfunc(nnTheta+perturb, X, y, lam, input_layer_size, hidden_layer_size, num_labels) \
                  - nncostfunc(nnTheta-perturb, X, y, lam, input_layer_size, hidden_layer_size, num_labels))/2/e
        perturb[i] = 0
    print('Neural Network, Numercial')
    print(np.concatenate([nngrad.reshape(-1,1), numgrad.reshape(-1,1)],axis=1))
    print('''If your backpropagation implementation is correct, then \r
          the relative difference will be small (less than 1e-9). \r
          Relative Difference: {0}'''.format(np.abs(np.mean(nngrad-numgrad)/np.mean(nngrad))))
    return

    
def predictnn(Theta1, Theta2, X):
    '''Predict the label for a trained Neural Network classifier'''
    m = X.shape[0]
    a1 = np.concatenate([np.ones([m, 1]), X], axis=1)
    a2 = np.zeros([m,Theta1.shape[0]+1])
    a2[:,0] = 1
    a2[:,1:] = sigmoid(a1@Theta1.T)
    a3 = np.zeros([m,Theta2.shape[0]])
    a3 = sigmoid(a2@Theta2.T)
    p = np.argmax(a3,axis=-1)+1
    return(p.reshape(-1,1))