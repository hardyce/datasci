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
from pybrain.tools.customxml import NetworkWriter
from pybrain.tools.customxml import NetworkReader
from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import diag, arange, meshgrid, where
from numpy.random import multivariate_normal
# Using pickle


def savenet(net):
    fileObject = open('C:\\Users\\hardy_000\\Documents\\datasci\\net', 'w')

    pickle.dump(net, fileObject)
    NetworkWriter.writeToFile(net, 'C:\\Users\\hardy_000\\Documents\\datasci\\net.xml')
    fileObject.close()
def loadnet(net):
    net = NetworkReader.readFrom('C:\\Users\\hardy_000\\Documents\\datasci\\net.xml')
    return net
#my_data = genfromtxt("C:\\Users\\hardy_000\\Downloads\\train.csv", delimiter=',')
#x=pd.read_csv("C:\\Users\\hardy_000\\Downloads\\train.csv")
def showpixel(i):
    
    b=np.reshape(x.iloc[i,1:],(-1,28))
    imshow(b,cmap='Greys')
    print "label " + str(x.loc[i,'label'])
def getPercentError(data):
    percentError(trainer.testOnClassData(dataset=data),data['class'])
def loaddata():
    z= pd.read_csv("C:\\Users\\hardy_000\\Downloads\\train.csv")
    pixels=z.drop('label',1)
    pixels=pixels.as_matrix()
    print len(pixels)
    targets=z['label'].as_matrix().reshape(-1,1)
    return targets, pixels


y,x=loaddata()
z=ds.classification.ClassificationDataSet(784,nb_classes=10)

z.setField('input', x)
z.setField('target', y)
#following linke seems to be necessary to fill class field of dataset
z._convertToOneOfMany( )
fnn = buildNetwork( z.indim, 300, z.outdim, outclass=SoftmaxLayer )
trainer = BackpropTrainer( fnn, dataset=z, momentum=0.1, 
                          verbose=True, weightdecay=0.01,learningrate=0.01,lrdecay=1)


