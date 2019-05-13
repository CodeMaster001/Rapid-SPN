#!/usr/bin/env python
'''
CREATED:2011-11-12 08:23:33 by Brian McFee <bmcfee@cs.ucsd.edu>

Spatial tree demo for matrix data
'''


import numpy
import sys
import os
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
from spn.gpu.TensorFlow import optimize_tf
from spn.structure.Base import Product, Sum, assign_ids, rebuild_scopes_bottom_up
from sklearn.metrics import accuracy_score
from numpy.random.mtrand import RandomState
from spn.algorithms.LearningWrappers import learn_parametric, learn_classifier
import urllib
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import fetch_openml
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from spn.gpu.TensorFlow import eval_tf
import time;
numpy.random.seed(42)


def  score(i):
	if i == 'g':
		return 0;
	else:
		return 1;

def one_hot(df,col):
	df = pd.get_dummies([col])
	df.drop()



credit,target = fetch_openml(name='cars', version=2,return_X_y=True)
theirs = list()
ours = list()
kf = KFold(n_splits=10,shuffle=True)
print(credit.shape)
ours_time_list = list();
theirs_time_list = list();
credit = numpy.nan_to_num(credit)
credit = credit.astype(numpy.float64)

for train_index, test_index in kf.split(credit):
	X = credit[train_index,:]
	#y_train=target[train_index]
	X = preprocessing.normalize(X, norm='l2')
	X_test = credit[test_index];	
	X_test = preprocessing.normalize(X_test, norm='l2')
	y_test=target[test_index]


	context = list()
	for i in range(0,X.shape[1]):
		context.append(Gaussian)





	ds_context = Context(parametric_types=context).add_domains(X)
	print("training normnal spm")
	
	theirs_time = time.time()
	spn_classification =  learn_parametric(numpy.array(X),ds_context,min_instances_slice=10)
	
	
	
	theirs_time = time.time()-theirs_time

	ll_original = log_likelihood(spn_classification, X)
	ll = log_likelihood(spn_classification, X)
	ll_test = log_likelihood(spn_classification,X_test)
	ll_test_original=ll_test[ll_test>-1000]




	print('Building tree...')
	original = time.time();
	T = SPNRPBuilder(data=numpy.array(X),ds_context=ds_context,target=X,prob=0.3,height=3,leaves_size=100)
	print("Building tree complete")
	
	T= T.build_spn();
	T.update_ids();
	from spn.io.Text import spn_to_str_equation
	spn = T.spn_node;
	ours_time = time.time()-original;
	ours_time_list.append(ours_time)
	ll = log_likelihood(spn, X)
	ll_test = log_likelihood(spn,X_test)
	ll_test=ll_test[ll_test>-1000]
	print(numpy.mean(ll_test_original))
	print(numpy.mean(ll_test))
	theirs.extend(ll_test_original)
	ours.extend(ll_test)
	theirs_time_list.append(theirs_time)

plot_spn(spn_classification, 'basicspn-original.png')
plot_spn(spn, 'basicspn.png')
print(theirs)
print(ours)
print(original)
print('---Time---')
print(numpy.mean(ours_time_list))
print(numpy.mean(theirs_time_list))
print('---ll---')
print(numpy.mean(ours))
print(numpy.mean(theirs))






