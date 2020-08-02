#!/usr/bin/env python
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

credit= fetch_openml(name='tic-tac-toe', version=1,return_X_y=True)[0]
credit = pd.DataFrame(data=credit)
credit = credit.apply(LabelEncoder().fit_transform)
print(credit.shape)
print(credit.head())
credit = credit.astype(float)
credit = numpy.nan_to_num(credit)
selector_array=[2,3,4]
np.save('selector',np.array(selector_array))
kf = KFold(n_splits=10,shuffle=True)
theirs = list()
ours = list()
ours_time_list = list()
theirs_time_list = list();
min_instances_slice=20
height=2
leaves_size=2

file_name='tic_tac_toe.10.log'
for train_index, test_index in kf.split(credit):
    X = credit[train_index,:]
    X=numpy.nan_to_num(X)
    #X = preprocessing.normalize(X, norm='l2')
    X_test = credit[test_index];
    X_test=numpy.nan_to_num(X_test)
    #X_test = preprocessing.normalize(X_test, norm='l2')
    #X = X.astype(numpy.float32)
    #X_test =X_test.astype(numpy.float32)
    context = list()
    for i in range(0,X.shape[1]):
        context.append(Categorical)





    ds_context = Context(parametric_types=context).add_domains(X)
    print("training normnal spm")
    theirs_time = time.time()
    spn_classification =  learn_parametric(numpy.array(X),ds_context,min_instances_slice=min_instances_slice)
    spn_time = time.time()-theirs_time


    #print(ll_test)
    spn_mean = numpy.mean(log_likelihood(spn_classification,X_test))



    print('Building tree...')
    original = time.time();
    T =  SPNRPBuilder(data=numpy.array(X),ds_context=ds_context,target=X,threshold=0.01,leaves_size=leaves_size,height=height,spill=0.75,selector_array=selector_array)


    T= T.build_spn();
    T.update_ids();
    from spn.io.Text import spn_to_str_equation
    spn = T.spn_node;
    print("Building tree complete")
    spnrp_time = time.time()-original;
    #plot_spn(spn,'spn_orig.png')
    #ll = log_likelihood(spn, X)
    #spn=optimize_tf(spn,X,epochs=10000,optimizer= tf.train.AdamOptimizer(0.001))
    spnrp_mean = numpy.mean(log_likelihood(spn,X_test))
    cmd=open('results/'+file_name+'.cmd','a')
    cmd.write(str(sys.argv)+'\n')
    cmd.flush();
    cmd.close();
    f=open('results/'+file_name,'a')
    #print(spnrp_mean)
    temp=str(spn_mean)+","+str(spnrp_mean)+","+str(spn_time)+","+str(spnrp_time)+","+str(min_instances_slice)+"\n"
    #temp=str(spnrp_mean)+","+str(spnrp_time)+","+str(min_instances_slice)+"\n"
    f.write(temp)
    f.flush()
    f.close()
