#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 11:39:06 2017

@author: mach_ju
"""
import numpy as np
import matplotlib.pyplot as plt
# import networkx as nx
import sample_to_plot_contour as cont
import printProgressBar as pb

def plotmean(labels, Z):
    #G = nx.Graph()
       
    
    mean_clusters = np.zeros((len(np.unique(labels)),Z.shape[1]))
    std_clusters = np.zeros((len(np.unique(labels)),Z.shape[1]))
    C = np.zeros((len(np.unique(labels))),dtype=list)
    pb.printProgressBar(0, len(np.unique(labels)), prefix='Mean and Contour:', length=40)
    i=0
    for lab in np.unique(labels):
        i += 1
        if lab != -1:
            idx = np.where(labels == lab)[0]
            mean_clusters[lab] = np.mean(Z[idx,:],axis=0)
            std_clusters[lab] = np.std(Z[idx,:],axis=0)
            C[lab] = np.cov(Z[idx].T)
#            plt.plot(mean_clusters[lab][0],mean_clusters[lab][1],'gx',markersize=18)
#            plt.annotate(lab, xy=(mean_clusters[lab][0],mean_clusters[lab][1]), xytext=(mean_clusters[lab][0]+1,mean_clusters[lab][1]+1))

            
            cont.cont(mean_clusters[lab], std_clusters[lab], C[lab])
                        
            #G.add_nodes_from((mean_clusters[lab][0],mean_clusters[lab][1]))
        pb.printProgressBar(i, len(np.unique(labels)), prefix='Mean and Contour:', length=40)
    #nx.draw(G)
    # plt.show()
    return mean_clusters, C
