#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import h5py
import time
import copy
from random import randint
#load MNIST data
MNIST_data = h5py.File('MNISTdata.hdf5', 'r')
x_train = np.float32(MNIST_data['x_train'][:] )
y_train = np.int32(np.array(MNIST_data['y_train'][:,0]))
x_test = np.float32( MNIST_data['x_test'][:] )
y_test = np.int32( np.array( MNIST_data['y_test'][:,0] ) )
MNIST_data.close()
#Implementation of stochastic gradient descent algorithm
#number of inputs
x_dim = 28
num_inputs = x_dim * x_dim
num_filter = 4
filter_dim = 3
map_dim = x_dim - filter_dim + 1
#number of outputs
num_outputs = 10
num_hidden = num_filter*map_dim**2
model = {}
model['W'] = np.random.randn(num_outputs,num_hidden) / np.sqrt(num_hidden)
model['b'] = np.random.randn(num_outputs) / np.sqrt(num_hidden)
model['K'] = np.random.randn(num_filter,filter_dim**2) / np.sqrt(filter_dim**2)

model_grads = copy.deepcopy(model)
def softmax_function(z):
    ZZ = np.exp(z)/np.sum(np.exp(z))
    return ZZ
def forward(x,y, model):
    x = x.reshape(x_dim,x_dim)
    x_map = []
    for i in range(map_dim):
        for j in range(map_dim):
            x_map.append(x[i:i+filter_dim,j:j+filter_dim].reshape(-1))
    
    Z = np.tensordot(model['K'],np.array(x_map).T,axes=1).reshape(-1)
    H = Z*(Z>0)
    U = np.dot(model['W'], H) + model['b']
    p = softmax_function(U)
    return p,H

def backward(x,y,p,H, model, model_grads):
    x = x.reshape(x_dim,x_dim)
    Y = np.zeros(10)
    Y[y] = 1
    dU = p - Y
    db = dU
    dW = np.outer(dU,H.T)
    delta = np.dot(model['W'].T,dU)
    filter_back = (delta * (H>0)).reshape(num_filter,map_dim**2)
    x_back = []
    for i in range(filter_dim):
        for j in range(filter_dim):
            x_back.append(x[i:i+map_dim,j:j+map_dim].reshape(-1))
    dK = np.tensordot(filter_back,np.array(x_back).T,axes=1)
    model_grads = {'W':dW, 'b':db, 'K':dK}
    return model_grads

import time
time1 = time.time()
LR = .01
num_epochs = 2
for epochs in range(num_epochs):
    #Learning rate schedule
    if (epochs > 5):
        LR = 0.001
    if (epochs > 10):
        LR = 0.0001
    if (epochs > 15):
        LR = 0.00001
    total_correct = 0
    for n in range( len(x_train)):
        n_random = randint(0,len(x_train)-1 )
        y = y_train[n_random]
        x = x_train[n_random][:]
        p,H= forward(x, y, model)
        prediction = np.argmax(p)
        if (prediction == y):
            total_correct += 1
        #print(p)
        model_grads = backward(x,y,p,H,model, model_grads)
        model['W'] = model['W'] - LR*model_grads['W']
        model['b'] = model['b'] - LR*model_grads['b']
        model['K'] = model['K'] - LR*model_grads['K']
    print(total_correct/np.float(len(x_train) ) )
time2 = time.time()
print(time2-time1)
######################################################
#test data
total_correct = 0
for n in range( len(x_test)):
    y = y_test[n]
    x = x_test[n][:]
    p,H = forward(x, y, model)
    prediction = np.argmax(p)
    if (prediction == y):
        total_correct += 1
print(total_correct/np.float(len(x_test) ) )

