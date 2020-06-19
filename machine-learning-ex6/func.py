# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 11:56:32 2020
Some functions to preform Regularized Linear Regression and Bias-Variance EXERSIZE 5
@author: Ji-Ping Dai
"""
import numpy as np
import matplotlib.pyplot as plt 
from scipy.optimize import minimize
from scipy.optimize import fmin_cg
from sklearn.svm import SVC
plt.rc('text',usetex=True)
plt.rc('font',family='Times New Roman')

def plotdata(X,y,ax):
    '''PLOTDATA Plots the data points x and y into a new figure'''
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax.plot(X[np.ravel(y==1),0],X[np.ravel(y==1),1],'kx')
    ax.plot(X[np.ravel(y==0),0],X[np.ravel(y==0),1],'ro')
    return

def dataset3Params(X,y,Xval,yval):
    '''returns the optimal choice of C and sigma '''
    C = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30])
    sigma = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30])
    m = len(C)
    error = np.zeros([m,m])
    for i in range(m):
        for j in range(m):
            model = SVC(kernel='rbf', C=C[i], gamma=1/2/sigma[j]**2)
            model.fit(X,y.ravel())
            error[i,j] = sum(model.predict(Xval) == yval.ravel())
    maxarg = np.unravel_index(np.argmax(error),error.shape)
    return C[maxarg[0]],sigma[maxarg[1]]

def processEmail(file_contents):
    '''preprocesses a the body of an email and returns a list of word_indices'''
    import pandas as pd
    import re
    from nltk import PorterStemmer
    stemmer = PorterStemmer()
    vocabList = pd.read_table('data/vocab.txt',header=None, names=['index'],index_col=1)
    word_indices = []
    file_contents = file_contents.lower()
    pattern = re.compile(r'\n')
    file_contents = pattern.sub(" ", file_contents)
    file_contents
    pattern = re.compile(r'[0-9]+')
    file_contents = pattern.sub("number", file_contents)
    pattern = re.compile(r'(http|https)://.*?\s')
    file_contents = pattern.sub("httpaddr", file_contents)
    file_contents
    pattern = re.compile(r'[^\s]+@[^\s]+')
    file_contents = pattern.sub("emailaddr", file_contents)
    pattern = re.compile(r'[$]+')
    file_contents = pattern.sub("dollar", file_contents)
    file_contents = file_contents.split(' ')
    for i in range(len(file_contents)):
        if file_contents[i].isalpha():
            words = stemmer.stem(file_contents[i])
            if words in vocabList.index:
                word_indices.append(int(vocabList.loc[words]))
    return word_indices
    