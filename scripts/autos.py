#!/usr/bin/env python
'''
CREATED:2011-11-12 08:23:33 by Brian McFee <bmcfee@cs.ucsd.edu>

Spatial tree demo for matrix data
'''


import numpy
import sys
sys.path.append('/Users/prajay/spnrp/spflow-spnrp/src')
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn import preprocessing
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
from spatialtree import *
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import fetch_openml
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from spn.gpu.TensorFlow import eval_tf,optimize_tf
from spn.structure.Base import *
import time;
import numpy as np, numpy.random
numpy.random.seed(42)
import logging



#tf.logging.set_verbosity(tf.logging.INFO)
logging.getLogger().setLevel(logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))



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




ours_time_list=list()
theirs_time_list=list()
theirs = list()
ours = list();
train_dataset_df,train_dataset_label = fetch_openml(name='autos', version=1,return_X_y=True)
train_dataset_df = pd.DataFrame(data=train_dataset_df)

kf = KFold(n_splits=10,shuffle=True)
file_name ='autos.log'
Categorical_index = [1,2,3,4,5,6,7,13,14,16,24,25]
min_instances_slice = 100
train_dataset_df = train_dataset_df.fillna(0)
for i in range(0,len(train_dataset_df.columns.values)):
    if i in Categorical_index:
        train_dataset_df[train_dataset_df.columns.values[i]] = pd.get_dummies(train_dataset_df[train_dataset_df.columns.values[i]])
for train_index, test_index in kf.split(train_dataset_df):
    X =train_dataset_df.values[train_index]
    X=X.astype(numpy.float32)
    X_test = train_dataset_df.values[test_index];
    X_test=X_test.astype(numpy.float32)

    print(X.shape)
    N = X.shape[0]
    D = X.shape[1]
    X_zero = X[X[:,-1]==0]


    context = list()
    Gaussian_index = [1,2,3,4,5,6,7,13,14,16,24,25]
    for i in range(0,X.shape[1]):
        if i in Gaussian_index:
            context.append(Categorical)
        else:
            context.append(Gaussian)



    ds_context = Context(parametric_types=context).add_domains(X)
    print("training normnal spn")
    theirs_time = time.time()
    spn_classification =   learn_parametric(X,ds_context)
    spn_time = time.time()-theirs_time
    spn_classification = optimize_tf(spn_classification,X,epochs=80000,optimizer= tf.train.AdamOptimizer(0.001)) 
    #tf.train.AdamOptimizer(1e-4))


    spn_mean = np.mean(eval_tf(spn_classification, X_test))
    #print(ll_test)
    #ll_test = log_likelihood(spn_classification,X_test)




    print('Building tree...')
    original = time.time();
    T =  SPNRPBuilder(data=X,ds_context=ds_context,target=X,leaves_size=2,height=3,samples_rp=20,prob=0.40)
    

    T= T.build_spn();
    T.update_ids();
    spn = T.spn_node;
    print("Building tree complete")
    spnrp_time = time.time()-original;
    
    spn=optimize_tf(spn,X,epochs=10000,optimizer= tf.train.AdamOptimizer(0.001))
    #ll_test = eval_tf(spn,X)
    spnrp_mean = np.mean(eval_tf(spn, X_test))
    f=open('results/'+file_name,'a')
    f.write(str(sys.argv)+"\n")
    #print(spnrp_mean)
    temp=str(spn_mean)+","+str(spnrp_mean)+","+str(spn_time)+","+str(spnrp_time)+","+str(min_instances_slice)+"\n"
    #temp=str(spnrp_mean)+","+str(spnrp_time)+","+str(min_instances_slice)+"\n"
    f.write(temp)
    f.flush()
    f.close()














