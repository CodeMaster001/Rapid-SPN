#!/usr/bin/env python


import numpy
import sys
import os
os.environ['NUMEXPR_MAX_THREADS'] = '16'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn import preprocessing
from sklearn.datasets import load_svmlight_file
from spatialtree import *;
from sklearn.model_selection import train_test_split
from libsvm.svmutil import *
from spn.structure.Base import Context
from spn.io.Graphics import plot_spn
from spn.algorithms.Sampling import sample_instances
from spn.structure.leaves.parametric.Parametric import Categorical, Gaussian
from spn.algorithms.Inference import log_likelihood
from spn.algorithms.splitting.RDC import get_split_cols_RDC_py, get_split_rows_RDC_py
from sklearn.datasets import load_iris,load_digits,fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from spn.algorithms.LearningWrappers import learn_parametric
from spn.gpu.TensorFlow import *
from spn.structure.Base import Product, Sum, assign_ids, rebuild_scopes_bottom_up
from sklearn.metrics import accuracy_score
from numpy.random.mtrand import RandomState
from spn.algorithms.LearningWrappers import learn_parametric, learn_classifier
from spn.algorithms.TransformStructure import Prune,Compress,SPN_Reshape
import urllib
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import fetch_openml
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from spn.gpu.TensorFlow import eval_tf
from spn.structure.Base import *
import time;
import numpy as np, numpy.random
numpy.random.seed(42)
import multiprocessing
import logging
import subprocess

dataset_name=sys.argv[1]
credit_train = pd.read_csv('../dataset/'+dataset_name+'.ts.data',delimiter=',')
credit_test = pd.read_csv('../dataset/'+dataset_name+'.test.data',delimiter=',')

def preprocess(X,X_test):
    X=numpy.nan_to_num(X)
    X = X.astype(numpy.float32)
    X_test=numpy.nan_to_num(X_test)
    X_test = X_test.astype(numpy.float32)
    return X,X_test

kf = KFold(n_splits=10,shuffle=True)
theirs = list()
ours = list()
ours_time_list = list()
theirs_time_list = list();
train_set = list()
test_set = list();
counter = 0;
context = list()


for i in range(0,credit_train.shape[1]):
    context.append(Categorical)
#parameters
output_file_name=dataset_name+'.log'
min_instances_slice=200000000
epochs=8000
height=int(sys.argv[2])
prob=0.4
leaves_size=15
threshold=0.4
opt_args= str(output_file_name) + ' ' + str(min_instances_slice) +' ' +str(epochs) + ' '+ str(height) + ' '+str(prob) + ' ' +str(leaves_size)+' '+str(threshold)+' 1.0 1'
X,X_test=preprocess(credit_train.values,credit_test.values)

np.save('train', X)
np.save("test",X_test)
np.save("context",context)
P=subprocess.Popen(['./experiment.py train.npy test.npy context.npy '+opt_args.strip()],shell=True)
P.communicate()
P.wait();
P.terminate()
print("process completed")


