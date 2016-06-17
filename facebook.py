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
import scipy as sp
from sklearn import decomposition
import sklearn as skl
from collections import Counter


fw = [500, 1000, 4, 3, 2, 10] 

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
    
def convertTime(df):
    df['x']=df['x']
    df['y']=df['y']
    
    initial_date = np.datetime64('2014-01-01T01:01', dtype='datetime64[m]')
    d_times = pd.DatetimeIndex(initial_date + np.timedelta64(int(mn), 'm') 
                               for mn in df.time.values)    
    df['hour'] = (d_times.hour+ d_times.minute/60)
    df['weekday'] = d_times.weekday 
    df['month'] = d_times.month 
    df['year'] = (d_times.year - 2013) 

    df = df.drop(['time'], axis=1) 
    return df
    
def scaleFeatures(df):
    df['x']=df['x']*fw[0]
    df['y']=df['y']*fw[1]
    df['hour']=df['hour']*fw[2]
    df['weekday']=df['weekday']*fw[3]
    df['month']=df['month']*fw[4]
    df['year']=df['year']*fw[5]
    return df
    
def getSquare(data,i,j,buff):
    
    data=data[((((j-1-buff)<=data['y']) & (data['y']<=j+buff))&(((i-1-buff)<=data['x']) & (data['x']<=i+buff)))]
    
    return data
    
def removeSquare(data,i,j):
    data=data[~((((j-1)<=data['y']) & (data['y']<=j))&(((i-1)<=data['x']) & (data['x']<=i)))]

    

    return data
def getPercentError(guess,label):
    correct=(guess==label)
    return (1./correct.shape[0])*sum(correct)
def plot3DScatter(data):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.xlabel('x')
    plt.ylabel('y')
    ax.set_zlabel('month')
    ax.scatter(data.x,data.y,data.time,c=data.place_id,marker='o',depthshade=False,lw = 0)
    plt.axis('equal')
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
    
def myf(a,w):
    lookupTable, indexed_dataSet = np.unique(a, return_inverse=True)
    
    y= np.bincount(indexed_dataSet,w)
    lookupTable[y.argsort()]
    res=(lookupTable[y.argsort()][::-1][:3])
    ret=np.empty((3))
    ret.fill(res[-1])
    ret[0:res.shape[0]]=res
    return ret
def trainData(X,y):
    X_train=X
    y_train=y

    #clf = neighbors.KNeighborsClassifier(25)
    clf = neighbors.KNeighborsClassifier(n_neighbors=25, weights='distance', 
                               metric='manhattan')
    ft=clf.fit(X_train, y_train)
    return ft
    
    

def predictTest(test,clf,y_train,X_train,X_train_acc):
    distances,indices=clf.kneighbors(test)
    knearest_locs=np.array(X_train)[indices]
    knearest_labels=np.array(y_train)[indices]
    knearest_acc=np.array(X_train_acc)[indices]
    points=np.array(test)
    reppoints=np.tile(points,(25,1,1))
    reppoints=reppoints.transpose((1,0,2))

    ty=knearest_locs-reppoints
    tys=np.square(ty)
    tyssum=np.sum(tys,axis=2)
    tyssums=np.sqrt(tyssum)
    
    tyssumsac=knearest_acc/knearest_acc


    
    result = np.empty_like(knearest_labels[:,0:3])
    for i,(x,y) in enumerate(zip(knearest_labels,tyssumsac)):
        result[i] = myf(x,y)
    return result

def predictRegion(train,test):

    #train['x']=normalize(train['x'])
    #train['y']=normalize(train['y'])
    #train['time']=normalize(train['time'])
    train['accuracy']=normalize(train['accuracy'])
    train=scaleFeatures(train)
    test=scaleFeatures(test)
    #test['x']=normalize(test['x'])
    #test['y']=normalize(test['y'])
    #test['time']=normalize(test['time'])


    y_train=train.place_id
    X_train=train.drop('place_id',1)
    X_train_acc=X_train.accuracy
    X_train=X_train.drop('accuracy',1)
    X_train=X_train.drop('row_id',1)
    ft=trainData(X_train,y_train)
    


    X_test=test.drop('accuracy',1)
    row_id=X_test['row_id']
    X_test=X_test.drop('row_id',1)



    pred=predictTest(X_test,ft,y_train,X_train,X_train_acc)
    result=np.insert(pred,0,row_id,axis=1)
    return result
    
numberOfPlaces=8

train=loadtraindata()
test=loadtestdata()
plt.figure(10, figsize=(14,16))
cmapm = plt.cm.viridis
cmapm.set_bad("0.5",1.)

    
for i in range(numberOfPlaces):
    sample_place=getSamplePlace(i)
    
    
    counts, binsX, binsY = np.histogram2d(sample_place["x"]*1000, sample_place["y"]*500, bins=100,weights=normalize(sample_place['accuracy']))
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

#kmeans starts here normalize data



#train=getSquare(train,1,1)
#train=getSquare(loadtraindata(),1,1,0.3)
#train, test = cross_validation.train_test_split(train, test_size=0.33, random_state=42)
train=convertTime(train)
test=convertTime(test)
s=train.place_id[~np.in1d(np.unique(np.where(train.year==1)),np.unique(np.where(train.year==2)))]
train=train[np.in1d(train.place_id,s)]




#train=train.drop('time',1)
#nd=normalize(small['accuracy'])
#small=small[nd<0.1]
p=0
labs=0
pred= np.empty((0,4), int)
for i in range(1,11):
    for j in range(1,11):
        temp=getSquare(test,i,j,0)
        res=predictRegion(getSquare(train,i,j,0.3),temp)
        test=removeSquare(test,i,j)
        pred=np.append(pred,res,0)
pred=pred[np.argsort(pred[:,0])]
labs=np.array(labs)[np.argsort(labs.index)]


re=pd.DataFrame(pred)
x=re[1].astype(str)+" "+re[2].astype(str)+" "+re[3].astype(str)
re=pd.DataFrame(np.concatenate((pred[:,0].reshape(-1,1),x.reshape(-1,1)),1))
re.columns=["row_id","place_id"]
re.to_csv("C:\\Users\\hardy_000\\fbcomp\\submission.csv",index=False)








