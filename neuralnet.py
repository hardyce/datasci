# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 17:55:19 2016

@author: hardy_000
"""

##base code taken from http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/

##modified to use sigmoid activation function and also to incorporate offset into weight matrices.

import numpy as np
import sklearn as sk
from sklearn import datasets
from sklearn import linear_model
import matplotlib.pyplot as plt
import math
import pandas as pd
from sklearn import utils
from itertools import izip






def calculate_loss(model):
    W1, W2 = model['W1'], model['W2']
    # Forward propagation to calculate our predictions
    a2,z2=propagateForward(X,W1)
    a3,z3=propagateForward(a2,W2)
    
    return (1./num_examples)*sum(sum((np.square(yvec.T-a3))))
    
    # Calculating the loss
def getGradients(x,y,W1,W2):
        #lossPrev=1000
        # Forward propagation
        a1=x
        
        y=y.T
        a2,z2=propagateForward(a1,W1)
        a3,z3=propagateForward(a2,W2)
        
            
            #if loss<lossPrev:
            #    epsilon=min(epsilon*1.001,0.05)
            #else:

            #    epsilon=epsilon*0.5

            #lossPrev=loss
        
            # Backpropagation
        delta3 = a3-y.T
        a2=np.insert(a2, 0, 1, axis=0)
        a1=np.insert(a1, 0, 1, axis=0)
        delta2=np.dot(W2.T,delta3)*(a2*(1.-a2))
        
        dW2=np.dot(delta3,a2.T)
        dW1=np.dot(delta2[1:],a1.T)
            # Add regularization terms (b1 and b2 don't have regularization terms)
        return dW1,dW2
                
def grouper(iterable, n):
    args = [iter(iterable)] * n
    return izip(*args)  
    
def predict(model, x):
    W1, W2= model['W1'], model['W2']
    # Forward propagation
    a2,z2=propagateForward(x.T,W1)
    a3,z3=propagateForward(a2,W2)
    
    return np.argmax(a3,axis=0)

def propagateForward(x,w):
        ##insert column of 1s
        
        z = np.dot(w,np.insert(x, 0, 1, axis=0))
        a = sigmoid(z)
        return a,z
# This function learns parameters for the neural network and returns the model.
# - nn_hdim: Number of nodes in the hidden layer
# - num_passes: Number of passes through the training data for gradient descent
# - print_loss: If True, print the loss every 1000 iterations
def build_model(X,nn_input_dim,nn_hdim,nn_output_dim, num_passes=1,miniBatchSize=5,epsilon=0.1):
    lossarray=np.zeros(shape=(1,2))
    epsilonInit=epsilon
    # Initialize the parameters to random values. We need to learn these.
    np.random.seed(3)
    W1 = 2*np.random.rand(nn_hdim,nn_input_dim+1)-1
    W2 = 2*np.random.rand(nn_output_dim,nn_hdim+1)-1
    #constant for annealing step size
    c=num_passes*(200/miniBatchSize)
    # This is what we return at the end
    model = {}
    
    iteration=0
    # Gradient descent. For each batch...
    for i in xrange(0, num_passes):
         
        trainingExamplesX,trainingExamplesy=utils.shuffle(X.T,yvec)
        
        for x,y in zip(grouper(trainingExamplesX,miniBatchSize),grouper(trainingExamplesy,miniBatchSize)):
            
            y=np.array(y)
            x=np.array(x)
            
            dW1,dW2=getGradients(x.T,y.T,W1,W2)
            dW2[:,1:] += reg_lambda * W2[:,1:]
            dW1[:,1:] += reg_lambda * W1[:,1:]
            # Gradient descent parameter update
            epsilon=epsilonInit/(1.+(iteration/c))
            W1 += -epsilon * dW1*(1./len(y))
            W2 += -epsilon * dW2*(1./len(y))
         
            # Assign new parameters to the model
            model = { 'W1': W1, 'W2': W2}
            iteration=iteration+1
            
            lossarray=np.vstack([lossarray,[iteration,calculate_loss(model)]])
            
            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
        
            
    axes = plt.gca()        
    axes.set_xlim([0,iteration])
    axes.set_ylim([0,1])
    plt.plot(lossarray[1:,0],lossarray[1:,1])
    plt.show()
    return model,lossarray
    
def plot_decision_boundary(pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = X.T[:, 0].min() - .5, X.T[:, 0].max() + .5
    y_min, y_max = X.T[:, 1].min() - .5, X.T[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.bone)
    plt.scatter(X.T[:, 0], X.T[:, 1], c=y.reshape(-1,1), cmap=plt.cm.bone)
    
def sigmoid(x):
  return 0.5*(np.tanh(0.5*x)+1)  
  
def loadtraindata():
    z= pd.read_csv("C:\\Users\\hardy_000\\Downloads\\train.csv")
    pixels=z.drop('label',1)
    pixels=pixels.as_matrix()
    pixels=pixels/255
    targets=z['label'].as_matrix().reshape(-1,1)
    return pixels,targets
def vectorize(y,numexamples):
    yvec=np.zeros(((numexamples,pd.unique(y).size)))
    for i in range(numexamples):
        yvec[i][y[i]]=1
    return yvec

np.random.seed(0)
X, y = sk.datasets.make_moons(200, noise=0.20)
#X,y=loadtraindata()
X=X.T
y=y.reshape(-1,1)
num_examples = len(X.T) # training set size
num_vars=len(X)#number of parameters
 # output layer dimensionality
#changes from single output to multicolumn
yvec=vectorize(y,num_examples)


    
# Gradient descent parameters (I picked these by hand)
 # learning rate for gradient descent
reg_lambda =0 # regularization strength
sigmoid=np.vectorize(sigmoid)
print "Building network"
print "input"
print num_vars
model,la = build_model(X,num_vars,300,nn_output_dim=2,miniBatchSize=5,epsilon=0.01)

#predict(model,np.array([-2,-2]))
#a,z=propagateForward(np.array([1,1]),np.array([1,1,-2]).T)
#a2,z2=propagateForward(a,model['W2'])
# Plot the decision boundary
#plot_decision_boundary(lambda x: predict(model, x))
#plt.title("Decision Boundary for hidden layer size 3")
# Train the logistic rgeression classifier

