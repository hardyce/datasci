# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import pandas as pd
from pylab import *
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from urllib import urlretrieve
import cPickle as pickle
import os
import gzip
import numpy as np
import pybrain.datasets as ds
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer

from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import diag, arange, meshgrid, where
from numpy.random import multivariate_normal


#my_data = genfromtxt("C:\\Users\\hardy_000\\Downloads\\train.csv", delimiter=',')
#x=pd.read_csv("C:\\Users\\hardy_000\\Downloads\\train.csv")
def showpixel(i):
    
    b=np.reshape(x.iloc[i,1:],(-1,28))
    imshow(b,cmap='Greys')
    print "label " + str(x.loc[i,'label'])
   
def loaddata():
    z= pd.read_csv("C:\\Users\\hardy_000\\Downloads\\train.csv")
    pixels=z.drop('label',1)
    pixels=pixels.as_matrix()
    print len(pixels)
    targets=z['label'].as_matrix().reshape(-1,1)
    return targets, pixels


y,x=loaddata()
z=ds.classification.ClassificationDataSet(784,nb_classes=10)

    #print array[:,:-1]
    #print array[:,-1]
    #datase11t.addSample(array[:,:-1], array[:,-1])
    #dataset.addSample(array[:,:-1], array[:,-2:-1])
z.setField('input', x)
z.setField('target', y)
fnn = buildNetwork( z.indim, 300, z.outdim, outclass=SoftmaxLayer )
trainer = BackpropTrainer( fnn, dataset=z, momentum=0.1, verbose=True, weightdecay=0.01)


trainer.trainEpochs( 1 )

for i in range(20):
        trainer.trainEpochs(1)
        trnresult = percentError(trainer.testOnClassData(), z['class'])
        
        print("epoch: %4d" % trainer.totalepochs,
                     "  train error: %5.2f%%" % trnresult)