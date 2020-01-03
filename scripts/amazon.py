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
from spatialtree import SPNRPBuilder
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
import logging

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
        node.weights=[0.1,0.2,0.4,0.3]
    else:
        pass;


def one_hot(df,col):
	df = pd.get_dummies([col])
	df.drop()











credit = fetch_openml(name='MiceProtein', version=1,return_X_y=True)[0]
credit = pd.DataFrame(credit)
credit = credit.replace(r'^\s+$', numpy.nan, regex=True)

credit = pd.DataFrame(data=credit)
theirs = list()
ours = list()

kf = KFold(n_splits=10,shuffle=True)
print(credit.head())
credit = credit.astype(np.float32)
credit = numpy.nan_to_num(credit)
theirs = list()
ours = list()
ours_time_list = list()
theirs_time_list = list();
for train_index, test_index in kf.split(credit):
    X = credit[train_index,:]
    X=numpy.nan_to_num(X)
    X = preprocessing.normalize(X, norm='l2')
    X_test = credit[test_index];	
    X_test = preprocessing.normalize(X_test, norm='l2')
    X = X.astype(numpy.float32)
    X_test =X_test.astype(numpy.float32)
    context = list()
    context.append(Gaussian)
    for i in range(0,X.shape[1]-1):
        context.append(Gaussian)

	



    ds_context = Context(parametric_types=context).add_domains(X)
    print("training normnal spm")
    theirs_time = time.time()
    #spn_classification =  learn_parametric(X,ds_context,threshold=0.3,min_instances_slice=5)
    theirs_time = time.time()-theirs_time
    #spn_classification = optimize_tf(spn_classification,X,epochs=10000,optimizer= tf.train.AdamOptimizer(0.001)) 
    #tf.train.AdamOptimizer(1e-4))


    #ll_test = eval_tf(spn_classification, X_test)
    #print(ll_test)
    #ll_test_original=ll_test




    print('Building tree...')
    original = time.time();
    T =  SPNRPBuilder(data=X,ds_context=ds_context,target=X,leaves_size=2,height=2,samples_rp=10,prob=0.8,spill=0.0)
    print("Building tree complete")
    

    T= T.build_spn();
    T.update_ids();
    from spn.io.Text import spn_to_str_equation
    spn = T.spn_node;
    ours_time = time.time()-original;
    ours_time_list.append(ours_time)
    #bfs(spn,print_prob)
    ll_test = log_likelihood(spn, X_test)
    
    
    #spn=optimize_tf(spn,X,epochs=10000,optimizer= tf.train.AdamOptimizer(0.001))
    #print(numpy.mean(ll_test_original))
    print(numpy.mean(ll_test))
    ours.append(numpy.mean(ll_test))
    theirs_time_list.append(theirs_time)
print(ours)

#plot_spn(spn_classification, 'basicspn-original.png')
#plot_spn(spn, 'basicspn.png')
print(numpy.mean(theirs_time_list))
print(numpy.var(theirs_time_list))
print(numpy.mean(ours_time_list))
print(numpy.var(ours_time_list))
numpy.savetxt('ours.time', ours_time_list, delimiter=',')
numpy.savetxt('theirs.time',theirs_time_list, delimiter=',')
numpy.savetxt('theirs.ll',theirs, delimiter=',')
numpy.savetxt('ours.ll',ours, delimiter=',')

print('---ll---')
print(numpy.mean(theirs))
print(numpy.var(theirs))

print(numpy.mean(ours))
print(numpy.var(ours))

if not os.path.exists("results/jura"):
    os.makedirs("results/jura")
#os.makedirs("results/madelon")
numpy.savetxt('results/jura/ours.time', ours_time_list, delimiter=',')
numpy.savetxt('results/jura/theirs.time',theirs_time_list, delimiter=',')
numpy.savetxt('results/jura/theirs.ll',theirs, delimiter=',')
numpy.savetxt('results/jura/ours.ll',ours, delimiter=',')










