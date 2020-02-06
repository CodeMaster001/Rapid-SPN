#!/usr/bin/env python
'''
CREATED:2011-11-12 08:23:33 by Brian McFee <bmcfee@cs.ucsd.edu>

Spatial tree demo for matrix data
'''


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
train_dataset,labels= fetch_openml(name='CIFAR_10', version=1,return_X_y=True)
train_dataset = pd.DataFrame(train_dataset).head(5000).values

X_train=train_dataset[:4000,:]
X_test=train_dataset[4000:,:]

# experiment.py train.csv test.csv context.npy instance_slice epochs height prob leaves_sizev

print(train_dataset.shape)
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
output_file_name='mnist.log'
min_instances_slice=30
epochs=8000
height=4
prob=0.5
leaves_size=15
threshold=0.4
bandwidth=0.2
predict_bandwidth=1

for i in range(0,train_dataset.shape[1]):
    context.append(Gaussian)
for j in [500,600,700,800,900,1000,1000,2000,3000,4000,5000,6000,70000]:
        min_instances_slice=j
        output_file_name='mnist_'+str(j)+'.log'  
        opt_args= str(output_file_name) + ' ' + str(min_instances_slice) +' ' +str(epochs) + ' '+ str(height) + ' '+str(prob) + ' ' +str(leaves_size)+' '+str(threshold) +' '+str(bandwidth)+' '+str(predict_bandwidth)
        X=numpy.nan_to_num(X_train)
        X = X.astype(numpy.float32)
        X = preprocessing.normalize(X, norm='l2') 
        X_test = numpy.nan_to_num(X_test)
        X_test = preprocessing.normalize(X_test, norm='l2')
        X = X.astype(numpy.float32)
        X_test =X_test.astype(numpy.float32)
        train_set.append(X)

        test_set.append(X_test)
        print("--")
        print(X.shape)
        print(X_test.shape)
        np.save('train', X)
        np.save("test",X_test)
        np.save("context",context)
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


