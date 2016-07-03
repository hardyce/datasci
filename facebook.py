# -*- coding: utf-8 -*-
"""
Created on Fri Jun 03 21:00:28 2016

@author: hardy_000
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import neighbors
from sklearn.neighbors import NearestNeighbors
from sklearn import cross_validation
import uuid
import scipy as sp
import os
import sklearn as skl

from sklearn import preprocessing
from bayes_opt import BayesianOptimization
import functools
fw = [0.6, 0.32935, 0.56515, 0.2670, 22, 52, 0.51785]
def processData(testing):
    if(testing):
        train, test = cross_validation.train_test_split(getSquare(loadtraindata(),1,1,1,0.3), test_size=0.33, random_state=42)
    else:
        train=loadtraindata()
        test=loadtestdata()

    train=featureEngineering(train)
    test=featureEngineering(test)
    train=removeNonRecentPlaces(train)
    #train=dropLowFreq(train,train.shape[0]*0.5)
    return train,test
def loadtraindata():
    z= pd.read_csv("C:\\Users\\hardy_000\\fbcomp\\train.csv")
    return z
    
def loadtestdata():
    z= pd.read_csv("C:\\Users\\hardy_000\\fbcomp\\test.csv")

    return z

def mapkprecision(truthvalues, predictions):
    '''
    This is a faster implementation of MAP@k valid for numpy arrays.
    It is only valid when there is one single truth value. 

    m ~ number of observations
    k ~ MAP at k -- in this case k should equal 3

    truthvalues.shape = (m,) 
    predictions.shape = (m, k)
    '''
    z = (predictions == truthvalues[:, None]).astype(np.float32)
    weights = 1./(np.arange(predictions.shape[1], dtype=np.float32) + 1.)
    z = z * weights[None, :]
    return np.mean(np.sum(z, axis=1))

    
def removeNonRecentPlaces(df):
    s=df.place_id[~np.in1d(np.unique(np.where(df.year==1)),np.unique(np.where(df.year==2)))]
    df=df[np.in1d(df.place_id,s)]
    return df
        
def prepareSubmission(pred):
    re=pd.DataFrame(pred)
    x=re[1].astype(str)+" "+re[2].astype(str)+" "+re[3].astype(str)
    re=pd.DataFrame(np.concatenate((pred[:,0].reshape(-1,1),x.reshape(-1,1)),1))
    re.columns=["row_id","place_id"]
    re.to_csv("submissionalex.csv",index=False)
def runSolution(testing,train,test,acc_w,daysin_w,daycos_w,minsin_w,mincos_w,weekdaysin_w,weekdaycos_w,x_w,y_w,year_w):
    if(testing):
        numSquares=1
        pred= np.empty((0,4), int)
        labels=np.empty((0,),int)
        for i in range(1,numSquares+1):
            for j in range(1,numSquares+1):

                testSquare=getSquare(test,i*cellwidth,j*cellwidth,cellwidth,0)
            if(testing):
                labels=np.append(labels,np.array(testSquare.place_id),0)
                testSquare=testSquare.drop('place_id',axis=1)
            
        

            res=predictRegion(getSquare(train,i*cellwidth,j*cellwidth,cellwidth,cellwidth*0.3),testSquare,acc_w,daysin_w,daycos_w,minsin_w,mincos_w,weekdaysin_w,weekdaycos_w,x_w,y_w,year_w)
            test=removeSquare(test,i*cellwidth,j*cellwidth,cellwidth)
            pred=np.append(pred,res,0)

    indexOrder=np.argsort(pred[:,0])

    pred=pred[indexOrder]

    if(testing==True):
        pred=mapkprecision(labels[indexOrder],pred[:,1])
        
    return pred
def featureEngineering(df):
    df['x']=df['x']
    df['y']=df['y']
    
    minute = 2*np.pi*((df["time"]//5)%288)/288
    df['minute_sin'] = (np.sin(minute)+1).round(4)
    df['minute_cos'] = (np.cos(minute)+1).round(4)
    del minute
    day = 2*np.pi*((df['time']//1440)%365)/365
    df['day_of_year_sin'] = (np.sin(day)+1).round(4)
    df['day_of_year_cos'] = (np.cos(day)+1).round(4)
    del day
    weekday = 2*np.pi*((df['time']//1440)%7)/7
    df['weekday_sin'] = (np.sin(weekday)+1).round(4)
    df['weekday_cos'] = (np.cos(weekday)+1).round(4)
    del weekday
    df['year'] = (((df['time'])//525600))
    df.drop(['time'], axis=1, inplace=True)
    df['accuracy'] = np.log10(df['accuracy'])
    return df
    
def scaleFeatures(df,acc_w,daysin_w,daycos_w,minsin_w,mincos_w,weekdaysin_w,weekdaycos_w,x_w,y_w,year_w):
    df['accuracy'] *= acc_w
    df['day_of_year_sin'] *= daysin_w
    df['day_of_year_cos'] *= daycos_w
    df['minute_sin'] *= minsin_w
    df['minute_cos'] *= mincos_w
    df['weekday_sin'] *= weekdaysin_w
    df['weekday_cos'] *= weekdaycos_w
    df.x *= x_w
    df.y *= y_w
    df['year'] *= year_w
    
    
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
    
    clf=trainNeighbors(X_train,y_train)

    return clf
    
    

def predictNeighbors(test,clf,y_train,X_train,X_train_acc):
    distances,indices=clf.kneighbors(test)
    knearest_locs=np.array(X_train)[indices]
    knearest_labels=np.array(y_train)[indices]
    knearest_acc=np.array(X_train_acc)[indices]
    points=np.array(test)
    reppoints=np.tile(points,(25,1,1))
    reppoints=reppoints.transpose((1,0,2))
    ty=knearest_locs-reppoints
    #with 0.565294588918
    #without 0.56522962995
    ty[:,3] = np.where(np.absolute(ty[:,3])>12, 24-np.absolute(ty[:,3]), ty[:,3])
    #ty[:,4] = np.where(ty[:,3]>3, 7-ty[:,4], ty[:,4])
    #ty[:,5] = np.where(ty[:,5]>6, 12-ty[:,5], ty[:,5])
    
    tys=np.square(ty)
    tyssum=np.sum(tys,axis=2)
    tyssums=np.sqrt(tyssum)
    
    tyssumsac=1./tyssums


    
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
def predictRegion(train,test,acc_w,daysin_w,daycos_w,minsin_w,mincos_w,weekdaysin_w,weekdaycos_w,x_w,y_w,year_w):

    #train['x']=normalize(train['x'])
    #train['y']=normalize(train['y'])
    #train['time']=normalize(train['time'])
    #train['accuracy']=normalize(train['accuracy'])
    train=scaleFeatures(train,acc_w,daysin_w,daycos_w,minsin_w,mincos_w,weekdaysin_w,weekdaycos_w,x_w,y_w,year_w)
    test=scaleFeatures(test,acc_w,daysin_w,daycos_w,minsin_w,mincos_w,weekdaysin_w,weekdaycos_w,x_w,y_w,year_w)
    #test['x']=normalize(test['x'])
    #test['y']=normalize(test['y'])
    #test['time']=normalize(test['time'])


    y_train=train.place_id
    X_train=train.drop('place_id',1)
    X_train_acc=X_train.accuracy
    #X_train=X_train.drop('accuracy',1)
    X_train=X_train.drop('row_id',1)
    print("training")
    ft=trainData(X_train,y_train)
    
    #X_test=test.drop('accuracy',1)
    X_test=test
    row_id=X_test['row_id']
    X_test=X_test.drop('row_id',1)


    print("Predicting")
    #pred=predictTest(X_test,ft,y_train,X_train,X_train_acc)
    pred=predictNeighbors(X_test,ft,y_train,X_train,X_train_acc)
    
    result=np.insert(pred,0,row_id,axis=1)
    return result
    





#kmeans starts here normalize data


uuid_string = str(uuid.uuid4())
#train=getSquare(train,1,1)
#train=getSquare(loadtraindata(),1,1,0.3)
numberOfNeighbors=25
testing=True
train,loadtest=processData(testing)


gridsize=10
cellwidth=10./gridsize


test=loadtest
numSquares=gridsize
pred=runSolution(testing,train=train,test=test,fw[0],fw[1],fw[1],fw[2],fw[2],fw[3],fw[3],fw[4],fw[5],fw[6])
fw = [0.6, 0.32935, 0.56515, 0.2670, 22, 52, 0.51785]

if not testing:
    prepareSubmission(pred)
else:
    print(pred)
f=functools.partial(runSolution,testing=True,train=train,test=test)
bo = BayesianOptimization(f=f,
                                  pbounds={
                                      'acc_w': (0, 1),
                                      
                                      # Fix w_y at 1000 as the most important feature
                                      #'w_y': (500, 2000), 
                                      "daysin_w": (0.1, 0.5),
                                      "daycos_w": (0.1, 0.5),
                                      "minsin_w": (0.2, 0.7),
                                      "mincos_w": (0.2, 0.7),
                                      "weekdaysin_w": (0, 0.4),
                                      "weekdaycos_w": (0, 0.4),
                                      "x_w": (18, 24),
                                      "y_w": (36, 49),
                                      "year_w": (0.4, 0.6),
                                      },
                                  verbose=True
                                  )
bo.maximize(init_points=2, n_iter=8, acq="ei", xi=0.1)#0,1 prefer exploration
with open(os.path.join('knn_params/{}.json'.format(uuid_string)), 'w+') as fh:
                fh.write(json.dumps(bo.res, sort_keys=True, indent=4))
