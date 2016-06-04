# -*- coding: utf-8 -*-
"""
Created on Fri Jun 03 21:00:28 2016

@author: hardy_000
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
def loadtraindata():
    z= pd.read_csv("C:\\Users\\hardy_000\\fbcomp\\train.csv")
    return z
    
def loadtestdata():
    z= pd.read_csv("C:\\Users\\hardy_000\\fbcomp\\test.csv")

    return z

train=loadtraindata()
places=train.groupby('place_id')
meanplaces=places.mean()
meanplaces=meanplaces.reset_index(drop=False)
#plot a few locations means
meanplaces[:100].plot(kind='scatter',x='x',y='y',c='place_id',colormap='RdYlGn')


numberOfRows=2
numberOfColumns=1
plt.figure(10, figsize=(14,16))
cmapm = plt.cm.viridis
cmapm.set_bad("0.5",1.)

for i in range(numberOfRows*numberOfColumns):
    sample_place=train[train.place_id==(train.place_id).unique()[i]]
    #sample_place.plot(ax=axes.reshape(-1)[i],kind='scatter',x='x',y='y',c='accuracy')
    
    counts, binsX, binsY = np.histogram2d(sample_place["x"], sample_place["y"], bins=100)
    extent = [binsX.min(),binsX.max(),binsY.min(),binsY.max()]
    plt.subplot(5,4,i+1)
    plt.imshow(np.log10(counts.T),
               interpolation='none',
               origin='lower',
               extent=extent,
               aspect="auto",
               cmap=cmapm)
    plt.grid(True, c='0.6', lw=0.5)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("pid: " + str(sample_place['place_id'])[0])
plt.show()