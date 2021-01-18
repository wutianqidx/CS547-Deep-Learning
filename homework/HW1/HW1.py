#!/usr/bin/env python
# coding: utf-8

# In[22]:


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
num_inputs = 28*28
#number of outputs
num_outputs = 10
num_hidden = 100
model = {}
model['W1'] = np.random.randn(num_hidden,num_inputs) / np.sqrt(num_inputs)
model['b1'] = np.random.randn(num_hidden) / np.sqrt(num_inputs)
model['W2'] = np.random.randn(num_outputs,num_hidden) / np.sqrt(num_hidden)
model['b2'] = np.random.randn(num_outputs) / np.sqrt(num_hidden)

model_grads = copy.deepcopy(model)
def softmax_function(z):
    ZZ = np.exp(z)/np.sum(np.exp(z))
    return ZZ
def forward(x,y, model):
    Z = np.dot(model['W1'], x) + model['b1']
    H = Z*(Z>0)
    U = np.dot(model['W2'], H) + model['b2']
    p = softmax_function(U)
    return p,H,Z

def backward(x,y,p,H,Z, model, model_grads):
    Y = np.zeros(10)
    Y[y] = 1
    dU = p - Y
    db2 = dU
    dW2 = np.outer(dU,H.T)
    delta = np.dot(model['W2'].T,dU)
    db1 = delta* (Z>0)
    dW1 = np.outer(db1,x.T)
    model_grads = {'W1':dW1, 'W2':dW2, 'b1':db1, 'b2':db2}
    return model_grads

import time
time1 = time.time()
LR = .01
num_epochs = 5
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
        p,H,Z = forward(x, y, model)
        prediction = np.argmax(p)
        if (prediction == y):
            total_correct += 1
        #print(p)
        model_grads = backward(x,y,p,H,Z,model, model_grads)
        model['W1'] = model['W1'] - LR*model_grads['W1']
        model['W2'] = model['W2'] - LR*model_grads['W2']
        model['b1'] = model['b1'] - LR*model_grads['b1']
        model['b2'] = model['b2'] - LR*model_grads['b2']
    print(total_correct/np.float(len(x_train) ) )
time2 = time.time()
print(time2-time1)
######################################################
#test data
total_correct = 0
for n in range( len(x_test)):
    y = y_test[n]
    x = x_test[n][:]
    p,H,Z = forward(x, y, model)
    prediction = np.argmax(p)
    if (prediction == y):
        total_correct += 1
print(total_correct/np.float(len(x_test) ) )


# In[ ]:




