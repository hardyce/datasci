# -*- coding: utf-8 -*-
"""
Created on Fri Jun 03 21:00:28 2016

@author: hardy_000
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

def loadtraindata():
    z= pd.read_csv("C:\\Users\\hardy_000\\fbcomp\\train.csv")
    return z
    
def loadtestdata():
    z= pd.read_csv("C:\\Users\\hardy_000\\fbcomp\\test.csv")

    return z
def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""
    # By Jake VanderPlas
    # License: BSD-style
    
    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)
    

#places=train.groupby('place_id')
#meanplaces=places.mean()
#meanplaces=meanplaces.reset_index(drop=False)
#plot a few locations means
#meanplaces[:100].plot(kind='scatter',x='x',y='y',c='place_id',colormap='RdYlGn')

def getSamplePlaces(numberOfPlaces=1):
        
    return train[train.place_id.isin((train.place_id).unique()[1:numberOfPlaces+1])]
def getSamplePlace(numberOfPlaces=1):
        
    return train[train.place_id==(train.place_id).unique()[numberOfPlaces]]

def normalize(series):
    return (series-min(series))/(max(series)-min(series))    

numberOfPlaces=8

train=loadtraindata()

plt.figure(10, figsize=(14,16))
cmapm = plt.cm.viridis
cmapm.set_bad("0.5",1.)

    
for i in range(numberOfPlaces):
    sample_place=getSamplePlace(i)
    
    
    counts, binsX, binsY = np.histogram2d(sample_place["x"], sample_place["y"], bins=100,weights=normalize(sample_place['accuracy']))
    extent = [0,10,0,10]
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


sample=getSamplePlaces(30)
norm = mpl.colors.Normalize(vmin=min(sample.place_id),vmax=max(sample.place_id))
cp=plt.cm.get_cmap('jet')
cp=discrete_cmap(30,cp)
cols=cp(norm(sample['place_id']))
cols[:,3]=normalize(sample['accuracy'])
fig=plt.figure(11, figsize=(14,16))
fig.add_subplot(111)
plt.scatter(sample['x'],sample['y'],color=cols)
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.xlabel('x')
plt.ylabel('y')
ax.set_zlabel('time')
ax.scatter(sample.x,sample.y,sample.time,c=cols,marker='o',depthshade=False,lw = 0)

