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
from spn.structure.Base import Product, Sum, assign_ids, rebuild_scopes_bottom_up
from sklearn.metrics import accuracy_score
from numpy.random.mtrand import RandomState
from spn.algorithms.LearningWrappers import learn_parametric, learn_classifier
from spn.algorithms.TransformStructure import Prune,Compress,SPN_Reshape
import urllib
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import fetch_openml
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
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



# experiment.py train.csv test.csv context.npy instance_slice epochs height prob leaves_size

train_dataset_df,labels= fetch_openml(name='vote', version=1,return_X_y=True)
le = preprocessing.LabelEncoder()
train_dataset_df = train_dataset_df[train_dataset_df.columns].apply(le.fit_transform)
print(train_dataset_df.head())
kf = KFold(n_splits=1,shuffle=True)
theirs = list()
ours = list()
ours_time_list = list()
theirs_time_list = list();
train_set = list()
test_set = list();
counter = 0;
context = list()

#parameters
output_file_name='vote.10.log'
min_instance_slice=40
epochs=8000
height=2
prob=0.4
leaves_size=15
threshold =0.4

for min_instance_slice in [10,20,40,50,100]:
    for height in [2,4,6,8,10,12,14,16,18,20,40,50,100,200,300,400]:
        for leaves_size in [2,4,6,8,10,12,14,16,18,24,48,72,200,300]:
            opt_args= str(output_file_name) + ' ' + str(min_instance_slice) + ' ' + str(height) +' '+ str(leaves_size) + ' ' +str(threshold)
            for i in range(0,train_dataset_df.shape[1]):
                context.append(Categorical)
            for train_index,test_index in kf.split(train_dataset_df):
                X_train,X_test=train_dataset_df.values[train_index],train_dataset_df.values[test_index]
                X=numpy.nan_to_num(X_train)
                X_test = X_test.astype(numpy.float32)
                X_test = numpy.nan_to_num(X_test)
                print(X_test.shape)
                train_set.append(X)
                test_set.append(X_test)
                np.save('train', X)
                np.save("test",X_test)
                np.save("context",context)
                P=subprocess.Popen(['python3 experiment.py train.npy test.npy context.npy '+opt_args.strip()],shell=True)
                P.communicate()
                P.wait();
                P.terminate()
            print("process completed")
