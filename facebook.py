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
from sklearn import neighbors, datasets
from sklearn.neighbors import NearestNeighbors
from sklearn import cross_validation
from sklearn.metrics import pairwise
from scipy.spatial import distance

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
    
def getSquare(data,i,j):
    data=data[(j-1)<data['y']]
    data=data[data['y']<j]
    data=data[(i-1)<data['x']]
    data=data[data['x']<i]
    return data

def plot3DScatter(data):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.xlabel('x')
    plt.ylabel('y')
    ax.set_zlabel('time')
    ax.scatter(normalize(data.x),normalize(data.y),normalize(data.time),c=data.place_id,marker='o',depthshade=False,lw = 0)
    plt.show()
#places=train.groupby('place_id')
#meanplaces=places.mean()
#meanplaces=meanplaces.reset_index(drop=False)
#plot a few locations means
#meanplaces[:100].plot(kind='scatter',x='x',y='y',c='place_id',colormap='RdYlGn')

def getSamplePlaces(data,numberOfPlaces=1):
        
    return data[data.place_id.isin((data.place_id).unique()[1:numberOfPlaces+1])]
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


sample=getSamplePlaces(train,30)
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
ax.scatter(normalize(sample.x),normalize(sample.y),normalize(sample.time),c=cols,marker='o',depthshade=False,lw = 0)
plt.show()

#kmeans starts here
train['time']=normalize(train['time'])
train['accuracy']=normalize(train['accuracy'])
train=getSquare(train,1,1)
X_train, X_test = cross_validation.train_test_split(train, test_size=0.33, random_state=42)
#train=train.drop('time',1)
#nd=normalize(small['accuracy'])
#small=small[nd<0.1]

y_train=X_train.place_id
X_train=X_train.drop('place_id',1)
X_train_acc=X_train.accuracy
X_train=X_train.drop('accuracy',1)
X_train=X_train.drop('row_id',1)
clf = neighbors.KNeighborsClassifier(5)

ft=clf.fit(X_train, y_train)
X_test_acc=X_test.accuracy
X_test=X_test.drop('accuracy',1)
X_test=X_test.drop('row_id',1)
y_test=X_test['place_id']
X_test=X_test.drop('place_id',1)
res=clf.predict(X_test)
bol=(res==y_test)
fin=(1./bol.size)*sum(bol)

##may want to link accuracy to below somehow
nbrs=NearestNeighbors(5,algorithm='ball_tree').fit(X_train)
distances,indices=nbrs.kneighbors(X_test)
knearest_locs=np.array(X_train)[indices]
knearest_labels=np.array(y_train)[indices]
knearest_acc=np.array(X_train_acc)[indices]
points=np.array(X_test)
reppoints=np.tile(points,(5,1,1))
reppoints=reppoints.transpose((1,0,2))

ty=knearest_locs-reppoints
tys=np.square(ty)
tyssum=np.sum(tys,axis=2)
tyssums=np.sqrt(tyssum)
tyssumsac=tyssums/knearest_acc
eh=np.apply_along_axis(np.bincount,1,np.arange(5),weights=tyssumsac)
np.bincount(np.arange(5),tyssumsac[0])