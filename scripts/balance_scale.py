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
        if node.weights[0]==node.weights[1]:
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


def libsvm_preprocess(file_name,row_size=62,colon_size=200):
    dataset_label, dataset_feature = svm_read_problem('../dataset/colon.csv')
    X=list()
    Y=list()
    for i in range(0,row_size):
        temp = list();
        for j in range(0,colon_size):
            temp.append(dataset_feature[i][j+1])
        X.append(temp)
        Y.append(dataset_label[i])
    return np.array(X),np.array(Y)
 

#print(credit.head())

def spnrp_train(X,X_test):
    context = list()
    for i in range(0,X.shape[1]):
        context.append(Gaussian)




    ds_context = Context(parametric_types=context).add_domains(X)
    original = time.time();
    T = SPNRPBuilder(data=numpy.array(X),ds_context=ds_context,target=X,prob=0.5,leaves_size=3,height=3,spill=0.3)
    print("Buiding tree complete")

    T= T.build_spn();
    T.update_ids();
    ours_time = time.time()-original;
    spn = T.spn_node;
    #spn=optimize_tf(spn,X,epochs=1000)
    #ll_test = eval_tf(spn,X_test)
    tf.reset_default_graph()
    del spn;
    return np.mean(ll_test),ours_time

def learnspn_train(X,X_test):
    context = list()
    for i in range(0,X.shape[1]):
        context.append(Gaussian)




    ds_context = Context(parametric_types=context).add_domains(X)
    theirs_time = time.time()
    spn_classification =  learn_parametric(numpy.array(X),ds_context,min_instances_slice=30)
    theirs_time = time.time()-theirs_time
    #spn_classification = optimize_tf(spn_classification,X,epochs=1000,optimizer= tf.train.AdamOptimizer(0.0001)) 
        #tf.train.AdamOptimizer(1e-4))


    #ll_test = eval_tf(spn_classification,X_test)
    #print(ll_test)
    ll_test = log_likelihood(spn_classifiation,X_test)
    ll_test_original=ll_test
    tf.reet_default_graph()
    del spn_classification
    return  np.mean(ll_test),theirs_time

# experiment.py train.csv test.csv context.npy instance_slice epochs height prob leaves_size
train_dataset,labels= fetch_openml(name='balance-scale', version=1,return_X_y=True)
train_dataset_df = pd.DataFrame(train_dataset)

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
epochs=8000
height=5
prob=0.4
leaves_size=20
threshold =0.4


for i in range(0,train_dataset_df.shape[1]):
    context.append(Gaussian)
for j in [15]:
    output_file_name='balance_'+str(j)+'.log'
    min_instances_slice=j
    opt_args= str(output_file_name) + ' ' + str(min_instances_slice) +' ' +str(epochs) + ' '+ str(height) + ' '+str(prob) + ' ' +str(leaves_size)+' ' + str(threshold)
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
        np.savetxt('train.csv', X, delimiter=',')
        np.savetxt("test.csv",X_test,delimiter=',')
        np.save("context",context)
        P=subprocess.Popen(['./experiment.py train.csv test.csv context.npy '+opt_args.strip()],shell=True)
        P.communicate()
        P.wait();
        P.terminate()
    print("process completed")
#!/usr/bin/env python
'''
CREATED:2011-11-12 08:23:33 by Brian McFee <bmcfee@cs.ucsd.edu>

Spatial tree demo for matrix data
'''


