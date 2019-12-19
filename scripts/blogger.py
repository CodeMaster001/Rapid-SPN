#!/usr/bin/env python
'''
CREATED:2011-11-12 08:23:33 by Brian McFee <bmcfee@cs.ucsd.edu>

Spatial tree demo for matrix data
'''
import logging
logger = logging.getLogger('spnrp')
# create file handler which logs even debug messages
logging.basicConfig(filename='spnrppx.log',level=logging.DEBUG)



import numpy
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random
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




#tf.logging.set_verbosity(tf.logging.INFO)
#logging.getLogger().setLevel(logging.INFO)
#logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

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





credit = fetch_openml(name='blogger', version=1,return_X_y=True)[0]
credit = pd.DataFrame(credit)
print(credit.head())
kf = KFold(n_splits=10,shuffle=True)
theirs = list()
ours = list()
ours_time_list = list()
theirs_time_list = list();
ours_time_tf = list()
theirs_time_tf = list();
for train_index, test_index in kf.split(credit):
    X = credit.values[train_index,:]
    print(X.shape)
    X=numpy.nan_to_num(X)
    #X = preprocessing.normalize(X, norm='l2')
    X_test = credit.values[test_index]; 
    X_test = numpy.nan_to_num(X_test)
    #X_test = preprocessing.normalize(X_test, norm='l2')
    X = X.astype(numpy.float32)
    X_test =X_test.astype(numpy.float32)
    context = list()
    for i in range(0,X.shape[1]):
       context.append(Categorical)


    ds_context = Context(parametric_types=context).add_domains(X)
    logging.info("training normnal spm")

    original = time.time()
    spn_classification =  learn_parametric(numpy.array(X),ds_context,min_instances_slice=10,threshold=0.7)

    
    #spn_classification = optimize_tf(spn_classification,X,epochs=1000,optimizer= tf.train.AdamOptimizer(0.001)) 
    #tf.train.AdamOptimizer(1e-4))

    theirs_time = time.time()-original


    #ll_test = eval_tf(spn_classification, X_test)
   # print(ll_test)
    ll_test = log_likelihood(spn_classification,X_test)
    theirs_time_tf = time.time() -original

    ll_test_original=ll_test


    logging.info('Building tree...')
    original = time.time();
    T = SPNRPBuilder(data=numpy.array(X),ds_context=ds_context,target=X,prob=0.6,leaves_size=2,height=2,spill=0.7)
    logging.info("Building tree complete")

    T= T.build_spn();
    T.update_ids();
    from spn.io.Text import spn_to_str_equation
    spn = T.spn_node;
    ours_time = time.time()-original
    ours_time_list.append(ours_time)
    #fs(spn,print_prob)
    
    ll_test = log_likelihood(spn, X_test)
    logging.info(np.mean(ll_test))
    spn=optimize_tf(spn,X,epochs=2000,batch_size=1000,optimizer= tf.train.AdamOptimizer(0.001))
    ll_test = eval_tf(spn,X_test)
    ours_time_tf = time.time()-original
    ll_test=ll_test
    logging.info("--ll--")
    logging.info(numpy.mean(ll_test_original))
    logging.info(numpy.mean(ll_test))
    logging.info(theirs_time)
    logging.info(ours_time)
    print(theirs_time_tf)
    print(ours_time_tf)
    theirs.append(numpy.mean(ll_test_original))
    ours.append(numpy.mean(ll_test))
    theirs_time_list.append(theirs_time)


    #plot_spn(spn_classification, 'basicspn-original.png')
#plot_spn(spn, 'basicspn.png')
logging.info(theirs)
logging.info(ours)
#print(original)
logging.info('---Time---')
logging.info(numpy.mean(theirs_time_list))
logging.info(numpy.var(theirs_time_list))
logging.info(numpy.mean(ours_time_list))
logging.info(numpy.var(ours_time_list))
logging.info('---ll---')
logging.info(numpy.mean(theirs))
logging.info(numpy.var(theirs))
logging.info(numpy.mean(ours))
logging.info(numpy.var(ours))

os.makedirs("results/blogger")
numpy.savetxt('results/blogger/ours.time', ours_time_list, delimiter=',')
numpy.savetxt('results/blogger/theirs.time',theirs_time_list, delimiter=',')
numpy.savetxt('results/blogger/theirs.ll',theirs, delimiter=',')
numpy.savetxt('results/blogger/ours.ll',ours, delimiter=',')

