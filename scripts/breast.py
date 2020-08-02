#!/usr/bin/env python

import numpy
import sys
import os
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
#tf.logging.set_verbosity(tf.logging.INFO)
logging.getLogger().setLevel(logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


train_dataset_data,labels= fetch_openml(name='breast-cancer', version=1,return_X_y=True)
train_dataset_df = pd.DataFrame(train_dataset_data)

kf = KFold(n_splits=10,shuffle=True)
train_set = list()
test_set = list();
counter = 0;
context = list()
min_instances_slice=8
epochs=8000
prob=0.6
threshold=0.2
height=6
leaves_size=12
output_file_name='breast.'+str(height)+'.'+str(leaves_size)+'.log'

opt_args= str(output_file_name) + ' ' + str(min_instances_slice) +' '+ str(height) + ' '+str(prob) + ' ' +str(leaves_size)+' '+str(threshold)

for i in range(0,train_dataset_df.shape[1]):
    context.append(Gaussian)
for train_index,test_index in kf.split(train_dataset_df):
    X_train,X_test=train_dataset_df.values[train_index],train_dataset_df.values[test_index]
    X=numpy.nan_to_num(X_train)
    X = X.astype(numpy.float32)
    X = preprocessing.normalize(X, norm='l2') 
    X_test = numpy.nan_to_num(X_test)
    X_test = preprocessing.normalize(X_test, norm='l2')
    X = X.astype(numpy.float32)
    X_test =X_test.astype(numpy.float32)
    train_set.append(X)
    test_set.append(X_test)
    np.save('train', X)
    np.save('test',X_test)
    np.save("context",context)
    P=subprocess.Popen(['./experiment.py train.npy test.npy context.npy '+opt_args.strip()],shell=True)
    P.communicate()
    P.wait();
    P.terminate()
    print("process completed")

