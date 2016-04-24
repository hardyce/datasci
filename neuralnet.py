# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 17:55:19 2016

@author: hardy_000
"""
import numpy as np
import sklearn as sk
from sklearn import datasets
import matplotlib.pyplot as plt
import math


np.random.seed(0)
X, y = sk.datasets.make_moons(200, noise=0.20)
X=X.T
y=y.reshape(-1,1)
num_examples = len(X.T) # training set size
nn_input_dim = 2 # input layer dimensionality
nn_output_dim = 1 # output layer dimensionality
 
# Gradient descent parameters (I picked these by hand)
epsilon = 0.01 # learning rate for gradient descent
reg_lambda = 0.01 # regularization strength





def calculate_loss(model):
    W1, W2 = model['W1'], model['W2']
    # Forward propagation to calculate our predictions
    a2,z2=propagateForward(X,W1)
    a3,z3=propagateForward(a2,W2)
    
    # Calculating the loss
    
    return (1./num_examples)*sum(sum((np.square(y.T-a3))))
    
def predict(model, x):
    W1, W2= model['W1'], model['W2']
    # Forward propagation
    a2,z2=propagateForward(x.T,W1)
    a3,z3=propagateForward(a2,W2)
   
    return np.around(a3, decimals=0)

def propagateForward(x,w):
        ##insert column of 1s
        x=np.insert(x, 0, 1, axis=0)
        z = np.dot(w,x)
        a = sigmoid(z)
        return a,z
# This function learns parameters for the neural network and returns the model.
# - nn_hdim: Number of nodes in the hidden layer
# - num_passes: Number of passes through the training data for gradient descent
# - print_loss: If True, print the loss every 1000 iterations
def build_model(nn_hdim, num_passes=20000, print_loss=False):
     
    # Initialize the parameters to random values. We need to learn these.
    np.random.seed(0)
    W1 = np.random.randn(nn_hdim,nn_input_dim+1) / np.sqrt(nn_input_dim)
    W2 = np.random.randn(nn_output_dim,nn_hdim+1) / np.sqrt(nn_hdim)


    # This is what we return at the end
    model = {}
     
    # Gradient descent. For each batch...
    for i in xrange(0, num_passes):
 
        # Forward propagation
        a1=X
        a2,z2=propagateForward(a1,W1)
        a3,z3=propagateForward(a2,W2)
 
        # Backpropagation
        delta3 = a3-y.T
        a2=np.insert(a2, 0, 1, axis=0)
        a1=np.insert(a1, 0, 1, axis=0)
        delta2=np.dot(W2.T,delta3)*a2*(1-a2)
        
        dW2=(1./num_examples)*(delta3).dot(a2.T)
        dW1=(1./num_examples)*(delta2[1:]).dot(a1.T)
        # Add regularization terms (b1 and b2 don't have regularization terms)
        dW2[:,1:] += reg_lambda * W2[:,1:]
        dW1[:,1:] += reg_lambda * W1[:,1:]
 
        # Gradient descent parameter update
        W1 += -epsilon * dW1
        W2 += -epsilon * dW2
         
        # Assign new parameters to the model
        model = { 'W1': W1, 'W2': W2}
         
        # Optionally print the loss.
        # This is expensive because it uses the whole dataset, so we don't want to do it too often.
        if print_loss and i % 1000 == 0:
          print "Loss after iteration %i: %f" %(i, calculate_loss(model))
     
    return model
    
def plot_decision_boundary(pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = X.T[:, 0].min() - .5, X.T[:, 0].max() + .5
    y_min, y_max = X.T[:, 1].min() - .5, X.T[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    print np.c_[xx.ravel(), yy.ravel()]
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X.T[:, 0], X.T[:, 1], c=y.reshape(-1), cmap=plt.cm.Spectral)
    
def sigmoid(x):
  return 1 / (1 + math.exp(-x))   

sigmoid=np.vectorize(sigmoid)
model = build_model(3, print_loss=True)
print model['W1']
print model['W2']
predict(model,np.array([2,0]))
# Plot the decision boundary
plot_decision_boundary(lambda x: predict(model, x))
plt.title("Decision Boundary for hidden layer size 3")
# Train the logistic rgeression classifier

