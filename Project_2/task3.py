# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 22:10:37 2018

@author: sai
"""
#reference:
#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_ml/py_kmeans/py_kmeans_understanding/py_kmeans_understanding.html
#

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
def ED(pt1,pt2):
    diff =0
    distance =0
    for i in range(len(pt1)):
        diff = np.power(pt1[i]-pt2[i],2)
        distance= distance+diff
    return np.sqrt(distance)

def update_cluster():
    global centroids
    global clusters
    for i in X:
        distances = [ED(centroids[j],i) for j in centroids]
        cluster_num = distances.index(min(distances))
        clusters[cluster_num+1].append(i)
def update_centroid():
    global centroids
    global visited
    global clusters
    for i in clusters:
        sum = np.sum(clusters[i],axis =0)
        new_centroid = sum/len(clusters[i])
        centroids[i] = new_centroid
      
def checkterm():
    global stop
    global centroids
    old_centroids = dict(centroids)
    for c in centroids:
        old = old_centroids[c]
        new = centroids[c]
        if np.sum((new-old)/old*100.0) > 0.1:
            stop = False
        if stop:
            break    

colors = 10*["g","r","c","b","k"]
X = np.array([[5.9, 3.2], [ 4.6, 2.9], [6.2, 2.8], [4.7, 3.2], [5.5, 4.2], [5.0, 3.0], [4.9, 3.1], [6.7, 3.1],[5.1,3.8], [6.0, 3.0]])
centroids = {1: [6.2, 3.2], 2: [6.6, 3.7], 3:[6.5, 3.0]} #initializing the centroids to the points given
clusters = {1: [], 2: [] ,3: []} #initialised the clusters to empty lists to store the points
stop = True
for i in range(10):
    update_cluster()
    update_centroid()
    checkterm()

plt.scatter(centroids[1][0], centroids[1][1],marker="o", color="r", s=50, linewidths=2)
plt.scatter(centroids[2][0], centroids[2][1],marker="o", color="g", s=50, linewidths=2)
plt.scatter(centroids[3][0], centroids[3][1],marker="o", color="b", s=50, linewidths=2)
plt.show()


plt.scatter(centroids[1][0], centroids[1][1],marker="o", color="r", s=50, linewidths=2)
plt.scatter(centroids[2][0], centroids[2][1],marker="o", color="g", s=50, linewidths=2)
plt.scatter(centroids[3][0], centroids[3][1],marker="o", color="b", s=50, linewidths=2)
for i in clusters[1]:
    plt.plot(i[0], i[1], '^', color='red', markersize=10, linewidth=4, markerfacecolor='white',markeredgecolor='red',markeredgewidth=1)  
for i in clusters[2]:
    plt.plot(i[0], i[1], '^', color='green', markersize=10, linewidth=4, markerfacecolor='white',markeredgecolor='green',markeredgewidth=1)  
for i in clusters[3]:
    plt.plot(i[0], i[1], '^', color='blue', markersize=10, linewidth=4, markerfacecolor='white',markeredgecolor='blue',markeredgewidth=1)  

plt.show()