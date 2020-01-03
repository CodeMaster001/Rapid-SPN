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
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, clone
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.metaestimators import if_delegate_has_method


N_SAMPLES = 5000
RANDOM_STATE = 42


class SPNEstimator(BaseEstimator):
    def __init__(self, data,spn_object=None,ds_context=None,leaves_size=8000,scope=None,threshold=0.4,prob=0.7,ohe=True,proportion=0.2,**kwargs):
        self.spn = SPNRPBuilder()

    def fit(self, X, y=None):


class print(__doc__)

logging.set_verbosity(tf.logging.INFO)
logging.getLogger().setLevel(logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
                   queue.append(c)





credit = fetch_openml(name='iris', version=1,return_X_y=True)[0]
credit = pd.DataFrame(credit)

kf = KFold(n_splits=10,shuffle=True)
theirs = list()
ours = list()
ours_time_list = list()
theirs_time_list = list();
for train_index, test_index in kf.split(credit):
    X = credit.values[train_index,:]
    X=numpy.nan_to_num(X)
    X = preprocessing.normalize(X, norm='l2')
    X_test = credit.values[test_index]; 
    #X_test = numpy.nan_to_num(X_test)
    X_test = preprocessing.normalize(X_test, norm='l2')
    X = X.astype(numpy.float32)
    X_test =X_test.astype(numpy.float32)
    context = list()
    for i in range(0,X.shape[1]):
        context.append(Gaussian)

    



    ds_context = Context(parametric_types=context).add_domains(X)
    print("training normnal spm")
    
    theirs_time = time.time()
    spn_classification =  learn_parametric(numpy.array(X),ds_context,min_instances_slice=20)
    theirs_time = time.time()-theirs_time
    #spn_classification = optimize_tf(spn_classification,X,epochs=1000,optimizer= tf.train.AdamOptimizer(0.0001)) 
        #tf.train.AdamOptimizer(1e-4))


    
    #ll_test = eval_tf(spn_classification, X_test)
    #print(ll_test)
    ll_test = log_likelihood(spn_classification,X_test)
    ll_test_original=ll_test



    print('Building tree...')
    original = time.time();
    T = SPNRPBuilder(data=numpy.array(X),ds_context=ds_context,target=X,prob=0.5,leaves_size=2,height=2,spill=0.0)
    print("Building tree complete")
    
    T= T.build_spn();
    T.update_ids();
    ours_time = time.time()-original;
    spn = T.spn_node;
    spn=optimize_tf(spn,X,epochs=60000,optimizer= tf.train.AdamOptimizer(0.0001))
    ll_test = eval_tf(spn,X_test)
    ll_test=ll_test
    print("--ll--")
    print(numpy.mean(ll_test_original))
    print(numpy.mean(ll_test))
    theirs.append(numpy.mean(ll_test_original))
    ours.append(numpy.mean(ll_test))
    theirs_time_list.append(theirs_time)
    ours_time_list.append(ours_time)
    from spn.algorithms.Statistics import get_structure_stats
    print(get_structure_stats(spn_classification))
    from spn.algorithms.Statistics import get_structure_stats
    print(get_structure_stats(spn))



#plot_spn(spn, 'basicspn.png')
print(theirs)
print(ours)
print(original)
print("---tt---")
print(numpy.mean(theirs_time_list))
print(numpy.var(theirs_time_list))
print(numpy.mean(ours_time_list))
print(numpy.var(ours_time_list))
print('---ll---')
print(numpy.mean(theirs))
print(numpy.var(theirs))

print(numpy.mean(ours))
print(numpy.var(ours))
os.makedirs("results/iris")
numpy.savetxt('results/iris/ours.time', ours_time_list, delimiter=',')
numpy.savetxt('results/iris/theirs.time',theirs_time_list, delimiter=',')
numpy.savetxt('results/iris/theirs.ll',theirs, delimiter=',')
numpy.savetxt('results/iris/ours.ll',ours, delimiter=',')





