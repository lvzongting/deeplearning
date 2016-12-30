import os
import dill
import pickle
#import tensorflow as tf
import scipy.io
import numpy as np
from itertools import *

'''
data, batch  = pickle.load(open('train.pkl'))
img,  labels = batch(data,50)
print img.shape, labels.shape
'''

train = True ; test = False
#train = False; test = True

if train: imgs   = scipy.io.loadmat('train_32x32.mat')['X']
if train: labels = scipy.io.loadmat('train_32x32.mat')['y']
if test:  imgs   = scipy.io.loadmat('test_32x32.mat' )['X']
if test:  labels = scipy.io.loadmat('test_32x32.mat' )['y']

imgs  = np.transpose(imgs,[3,0,1,2])
data  = [imgs,labels]
batch = lambda x,y:[np.asarray(list(islice(cycle(x[0]),0,y))),
                    np.asarray(list(islice(cycle(x[1]),0,y)))]                        
print('start dump')
if train: pickle.dump([data,batch],open('train.pkl','w'))
if test:  pickle.dump([data,batch],open('test.pkl' ,'w'))
print('Done')
