#!/usr/bin/env python
'''
CREATED:2011-11-12 08:23:33 by Brian McFee <bmcfee@cs.ucsd.edu>

Spatial tree demo for matrix data
'''


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
from sklearn.feature_extraction.text import TfidfVectorizer
import time;
import numpy as np, numpy.random
import multiprocessing
import logging
from sklearn.datasets import load_digits
import subprocess
from sklearn.datasets import fetch_20newsgroups
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



print('Command Line Parameters:rows columns ratio repeat instance_slice threshold height leaves_size')

rows_size=int(sys.argv[1])
columns_size=int(sys.argv[2])
ratio = float(sys.argv[3])
repeat=int(sys.argv[4])
instance_slice=int(sys.argv[5])
threshold=float(sys.argv[6])
height=int(sys.argv[7])
leaves_size=int(sys.argv[8])
print(sys.argv)
 # experiment.py train.csv test.csv context.npy instance_slice epochs height prob leaves_size
train_dataset,labels= fetch_openml(name='CIFAR_10', version=1,return_X_y=True)
train_dataset_pd = pd.DataFrame(train_dataset).copy();

for i in range(0,repeat):

    print(train_dataset_pd.values.shape)

    train_dataset=np.array(train_dataset_pd.sample(n=int(rows_size)).values)
    train_dataset=train_dataset[:,:int(columns_size)]
    X_train,X_test=train_test_split(train_dataset,test_size=ratio)
    print(X_train.shape)
    print(X_test.shape)
    # experiment.py train.csv test.csv context.npy instance_slice epochs height prob leaves_sizev

    output_file_name='mnist_spnrp_'+str(columns_size)+'.log'
    epochs=8000
    selector_array=[2,3,4]
    np.save('selector',np.array(selector_array))
    context = list()

    for j in range(0,train_dataset.shape[1]):
        context.append(Gaussian)

    X=numpy.nan_to_num(X_train)
    X = X.astype(numpy.float32)
    X = preprocessing.normalize(X, norm='l2') 
    X_test = numpy.nan_to_num(X_test)
    X_test = preprocessing.normalize(X_test, norm='l2')
    X = X.astype(numpy.float32)
    X_test =X_test.astype(numpy.float32)
    print("--")
    print(X.shape)
    print(X_test.shape)
    np.save('train', X)
    np.save("test",X_test)
    np.save("context",context)
    opt_args= str(output_file_name) + ' ' + str(instance_slice) +' ' +str(height) + ' '+str(leaves_size)+' '+str(threshold) 
    P=subprocess.Popen(['./experiment.py train.npy test.npy context.npy '+opt_args.strip()],shell=True)
    P.communicate()
    P.wait();
    P.terminate()
    print("process completed")

#!/usr/bin/env python
'''
CREATED:2011-11-12 08:23:33 by Brian McFee <bmcfee@cs.ucsd.edu>

Spatial tree demo for matrix data
'''


