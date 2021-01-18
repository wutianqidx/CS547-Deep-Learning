import numpy as np
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.distributed as dist
import torchvision

from helperFunctions import getUCF101
#from helperFunctions import loadSequence
#import resnet_3d

import h5py
#import cv2

from multiprocessing import Pool


IMAGE_SIZE = 224
NUM_CLASSES = 101
batch_size = 100
lr = 0.0001
num_of_epochs = 10


data_directory = '/projects/training/bayw/hdf5/'
class_list, train, test = getUCF101(base_directory = data_directory)

single_frame_dir = 'UCF-101-predictions/'
seq_dir = 'UCF-101-predictions_3d/'

acc_top1 = 0.0
acc_top5 = 0.0
acc_top10 = 0.0
confusion_matrix = np.zeros((NUM_CLASSES,NUM_CLASSES),dtype=np.float32)
random_indices = np.random.permutation(len(test[0]))

for i in range(len(test[0])):
    index = random_indices[i]
    t1 = time.time()

    filename = test[0][index]
    filename = filename.replace('.avi','.hdf5')
    filename = filename.replace('UCF-101','UCF-101-hdf5')
    h = h5py.File(filename,'r')
    nFrames = len(h['video'])
    sf_filename = filename.replace(data_directory+'UCF-101-hdf5/',single_frame_dir)
    seq_filename = filename.replace(data_directory+'UCF-101-hdf5/',seq_dir)
    sf_pred = (h5py.File(sf_filename,'r')).get('predictions').value
    seq_pred = (h5py.File(seq_filename,'r')).get('predictions').value
    
    # softmax
    #for j in range(sf_pred.shape[0]):
    #    sf_pred[j] = np.exp(sf_pred[j])/np.sum(np.exp(sf_pred[j]))

    #for j in range(seq_pred.shape[0]):
    #    seq_pred[j] = np.exp(seq_pred[j])/np.sum(np.exp(seq_pred[j]))


    # concat
    prediction = np.concatenate((sf_pred,seq_pred),axis=0)
    for j in range(prediction.shape[0]):
        prediction[j] = np.exp(prediction[j])/np.sum(np.exp(prediction[j]))
    prediction = np.sum(np.log(prediction),axis=0)

    #prediction = np.sum(np.log(sf_pred),axis=0) + np.sum(np.log(seq_pred),axis=0)
    #print(prediction,prediction/2)
    argsort_pred = np.argsort(-prediction)[0:10]


    label = test[1][index]
    confusion_matrix[label,argsort_pred[0]] += 1
    if(label==argsort_pred[0]):
        acc_top1 += 1.0
    if(np.any(argsort_pred[0:5]==label)):
        acc_top5 += 1.0
    if(np.any(argsort_pred[:]==label)):
        acc_top10 += 1.0

    print('i:%d nFrames:%d t:%f (%f,%f,%f)' 
          % (i,nFrames,time.time()-t1,acc_top1/(i+1),acc_top5/(i+1), acc_top10/(i+1)))


number_of_examples = np.sum(confusion_matrix,axis=1)
for i in range(NUM_CLASSES):
    confusion_matrix[i,:] = confusion_matrix[i,:]/np.sum(confusion_matrix[i,:])

results = np.diag(confusion_matrix)
indices = np.argsort(results)

sorted_list = np.asarray(class_list)
sorted_list = sorted_list[indices]
sorted_results = results[indices]

for i in range(NUM_CLASSES):
    print(sorted_list[i],sorted_results[i],number_of_examples[indices[i]])

np.save('combine_confusion_matrix_concat.npy',confusion_matrix)








