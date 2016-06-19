# -*- coding: utf-8 -*-
"""
Created on Fri Jun 03 21:00:28 2016

@author: hardy_000
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib as mpl
import xgboost as xgb
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
from sklearn import preprocessing


fw = [500, 1000, 4, 3, 2, 10] 

def loadtraindata():
    z= pd.read_csv("C:\\Users\\hardy_000\\fbcomp\\train.csv")
    return z
    
def loadtestdata():
    z= pd.read_csv("C:\\Users\\hardy_000\\fbcomp\\test.csv")

    return z

    

    
def removeNonRecentPlaces(df):
    s=df.place_id[~np.in1d(np.unique(np.where(df.year==1)),np.unique(np.where(df.year==2)))]
    df=df[np.in1d(df.place_id,s)]
    return df
        
    
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
    
def getSquare(data,i,j,width,buff):
    
    data=data[((((j-width-buff)<=data['y']) & (data['y']<=j+buff))&(((i-width-buff)<=data['x']) & (data['x']<=i+buff)))]
    
    return data
    
def removeSquare(data,i,j,width):
    data=data[~((((j-width)<=data['y']) & (data['y']<=j))&(((i-width)<=data['x']) & (data['x']<=i)))]

    

    return data
def getPercentError(guess,label):
    correct=(guess==label)
    return (1./correct.shape[0])*sum(correct)

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
    
def trainNeighbors(data,labels):

    clf = neighbors.KNeighborsClassifier(25)
    clf = neighbors.KNeighborsClassifier(n_neighbors=25, weights='distance', 
                              metric='manhattan')
    return clf.fit(data,labels)
    
def trainXGB(X_train,y_train):
    
    le=preprocessing.LabelEncoder()
    le.fit(y_train)
    labels=le.transform(y_train)
    dm_train = xgb.DMatrix(X_train, label=labels)
    
    res = xgb.cv({'eta': 0.1, 'objective': 'multi:softprob',
             'num_class': len(le.classes_),
             'alpha': 0.1, 'lambda': 0.1, 'booster': 'gbtree','nthreads':4},
            dm_train, num_boost_round=200, nfold=3, seed=42,
            early_stopping_rounds=10
            
        )
    N_epochs = res.shape[0]
    param = {'nthreads':4,'early_stopping_rounds':10, 'verbose_eval':10,'max_depth':3,'eta':0.1, 'silent':1, 'objective':'multi:softprob','num_class': len(np.unique(labels)) }
    #clf=trainNeighbors(X_train,labels)

    clf=xgb.train(param,dm_train,num_boost_round=N_epochs)
    return clf,le
    
def trainData(X,y):
    X_train=X
    y_train=y
    
    clf,le=trainXGB(X_train,y_train)

    return clf,le
    
    

def predictNeighbors(test,clf,y_train,X_train,X_train_acc):
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
def predictXGB(ft,data,le):
    pred=ft.predict(xgb.DMatrix(data))
    pred_idx = np.argsort(pred, axis=1)[:, -3:][:, ::-1]
    c = np.array(le.classes_)
    pred_id = np.take(c, pred_idx)
    return pred_id
def dropLowFreq(data,keepN):
    vc = data.place_id.value_counts()

# eliminate all ids which are low enough frequency
    vc = vc[np.cumsum(vc.values) < keepN]
    df1 = pd.DataFrame({'place_id': vc.index, 'freq': vc.values})

# this represents the training set after all low frequency place_ids
# are removed
    return pd.merge(data, df1, on='place_id',how='inner').drop('freq',axis=1)
def predictRegion(train,test):

    #train['x']=normalize(train['x'])
    #train['y']=normalize(train['y'])
    #train['time']=normalize(train['time'])
    train['accuracy']=normalize(train['accuracy'])
    #train=scaleFeatures(train)
    #test=scaleFeatures(test)
    #test['x']=normalize(test['x'])
    #test['y']=normalize(test['y'])
    #test['time']=normalize(test['time'])


    y_train=train.place_id
    X_train=train.drop('place_id',1)
    X_train_acc=X_train.accuracy
    X_train=X_train.drop('accuracy',1)
    X_train=X_train.drop('row_id',1)
    print("training")
    ft,le=trainData(X_train,y_train)
    


    X_test=test.drop('accuracy',1)
    row_id=X_test['row_id']
    X_test=X_test.drop('row_id',1)


    print("Predicting")
    #pred=predictTest(X_test,ft,y_train,X_train,X_train_acc)
    pred=predictXGB(ft,X_test,le)
    
    result=np.insert(pred,0,row_id,axis=1)
    return result
    





#kmeans starts here normalize data



#train=getSquare(train,1,1)
#train=getSquare(loadtraindata(),1,1,0.3)
numberOfNeighbors=25
testing=False
if(testing):
    train, test = cross_validation.train_test_split(getSquare(loadtraindata(),1,1,1,0.3), test_size=0.33, random_state=42)
else:
    train=loadtraindata()
    test=loadtestdata()

#train=convertTime(train)
#test=convertTime(test)
#train=removeNonRecentPlaces(train)
train=dropLowFreq(train,train.shape[0]*0.5)

gridsize=50
cellwidth=10./gridsize




#train=train.drop('time',1)
#nd=normalize(small['accuracy'])
#small=small[nd<0.1]
numSquares=gridsize
if(testing):
    numSquares=1
pred= np.empty((0,4), int)
labels=np.empty((0,),int)
for i in range(1,numSquares+1):
    for j in range(1,numSquares+1):
        print i
        print j
        testSquare=getSquare(test,i*cellwidth,j*cellwidth,cellwidth,0)
        if(testing):
            labels=np.append(labels,np.array(testSquare.place_id),0)
            testSquare=testSquare.drop('place_id',axis=1)
            
        

        res=predictRegion(getSquare(train,i*cellwidth,j*cellwidth,cellwidth,0),testSquare)
        test=removeSquare(test,i*cellwidth,j*cellwidth,cellwidth)
        pred=np.append(pred,res,0)

indexOrder=np.argsort(pred[:,0])

pred=pred[indexOrder]
if(testing==True):
    print(getPercentError(pred[:,1],labels[indexOrder]))



re=pd.DataFrame(pred)
x=re[1].astype(str)+" "+re[2].astype(str)+" "+re[3].astype(str)
re=pd.DataFrame(np.concatenate((pred[:,0].reshape(-1,1),x.reshape(-1,1)),1))
re.columns=["row_id","place_id"]
re.to_csv("C:\\Users\\hardy_000\\fbcomp\\submission.csv",index=False)
data = np.random.rand(5,10) # 5 entities, each contains 10 features
label = np.random.randint(2, size=5) # binary target
dtrain = xgb.DMatrix( data, label=label)

