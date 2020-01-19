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
#tf.logging.set_verbosity(tf.logging.INFO)
logging.getLogger().setLevel(logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

def bfs(root, func):
    seen, queue = set([root]), collections.deque([root])
    while queue:
        node = queue.popleft()
        func(node)
        if not isinstance(node, Leaf):
            for c in node.children:
                if c not in seen:
                    seen.add(c)
                    queue.append(c)

def print_prob(node):
    if isinstance(node,Sum):
        node.weights= np.random.dirichlet(np.ones(len(node.weights)),size=1)[0]
def  score(i):
    if i == 'g':
        return 0;
    else:
        return 1;

def one_hot(df,col):
    df = pd.get_dummies([col])
    df.drop()


def clean_data(x):
    try:
        return str(x).split(':')[-1]
    except:
        print(str(x))

 

# experiment.py train.csv test.csv context.npy instance_slice epochs height prob leaves_size

train_dataset_df = fetch_openml(name='bodyfat', version=1,return_X_y=True)[0]
train_dataset_df = pd.DataFrame(data=train_dataset_df)
train_dataset_df = train_dataset_df.fillna(0)


kf = KFold(n_splits=10,shuffle=True)
theirs = list()
ours = list()
ours_time_list = list()
theirs_time_list = list();
train_set = list()
test_set = list();
counter = 0;
context = list()

#parameters
output_file_name='bodyfat.log'
min_instances_slice=25
epochs=8000
height=4
prob=0.6
leaves_size=2
threshold=0.4

opt_args= str(output_file_name) + ' ' + str(min_instances_slice) +' ' +str(epochs) + ' '+ str(height) + ' '+str(prob) + ' ' +str(leaves_size)+' '+str(threshold)

for i in range(0,train_dataset_df.shape[1]):
    context.append(Gaussian)
for train_index,test_index in kf.split(train_dataset_df):
    X = train_dataset_df.values[train_index]
    X=numpy.nan_to_num(X)
    X = X.astype(numpy.float32)
    X = preprocessing.normalize(X, norm='l2')
    X_test = train_dataset_df.values[test_index]; 
    X_test = numpy.nan_to_num(X_test)
    X_test = preprocessing.normalize(X_test, norm='l2')
    X = X.astype(numpy.float32)
    X_test =X_test.astype(numpy.float32)
    context = list()
    for i in range(0,X.shape[1]):
        context.append(Gaussian)

    
    train_set.append(X)
    test_set.append(X_test)
    np.savetxt('train.csv', X,delimiter=',')
    np.savetxt("test.csv",X_test,delimiter=',')
    np.save("context",context)
    P=subprocess.Popen(['./experiment.py train.csv test.csv context.npy '+opt_args.strip()],shell=True)
    P.communicate()
    P.wait();
    P.terminate()
print("process completed")


