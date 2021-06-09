#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 13:25:37 2017

@author: mach_ju
"""

import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

def R(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])

def cont(mu, sigmas, C):
    ### construct covariance (diagonal at this stage)
    #C = np.diag(sigmas**2)
    C = C
    
    ### compute level (value) of contour line of the one sigma value, i.e., x=1*sigma
    ### we will need this for plotting later


    ### first: construct gaussian from mu and Cov
    G = multivariate_normal(mu,C)
    ### compute level by evaluating G at the sigma (around the mean)
    level = [G.pdf(sigmas+mu)]
   
    ### we introduce some degree for rotating the distribution (to introduce some non-zero non-diagonal elements)
#    theta = np.deg2rad(45)
#    ### rotate the covariance
#    ### this is how your general covariance matrix of a node will look like
#    C_rot = np.dot(R(theta),np.dot(C,R(theta).T))
    C_rot=C
    ### let's draw everything
    ### we need a grid NxN
    N = 151
    ### get the diagonal elements of the rotated covariance
    sig_rot = np.sqrt(np.diag(C_rot))
    ### compute grid in x and y (aligned to the old - non-rotated - domain)
    x = np.linspace(mu[0]-5*sig_rot[0],mu[0]+5*sig_rot[0],N)
    y = np.linspace(mu[1]-5*sig_rot[1],mu[1]+5*sig_rot[1],N)
    ### we need a meshgrid for plotting later a 2D function
    X, Y = np.meshgrid(x,y)  
    ### define and compute the 2D function
    GContour = np.zeros(X.shape)
    for i in range(N):
        for j in range(N):
            ### we need to evaluate the Gaussian at each grid point
            ### in fact we compute the squared Mahalanobis distance again
            r = np.subtract(np.array([x[i],y[j]]).reshape((2,1)),mu.reshape((2,1)))
            ### plug the values into the definition of a Gaussian
            GContour[i,j] = 1 / (2*np.pi*np.sqrt(np.linalg.linalg.det(C_rot))) * np.exp(-.5*np.dot(r.T,np.dot(np.linalg.linalg.inv(C_rot),r)))
    
#    ### let's draw some samples from the distribution (this could be your cluster)
#    draw_samples = [np.random.multivariate_normal(mu, C_rot) for i in range(151)]
#    ### plot the mean
    plt.plot(mu[0],mu[1],'x',color='#FF694F',markersize=10,alpha=1)
#    ### plot the samples of the distribution
#    draw_samples= np.array(draw_samples).T
#    #plt.plot(draw_samples[0,:],draw_samples[1,:],'b.')
    
    ### plot the contour of the Gaussian at the 1 sigma level
    plt.contour(X,Y,GContour,1,levels=level,colors=('#FF694F'),linewidths=3,alpha=1.0)
    plt.axis('equal')
