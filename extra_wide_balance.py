#!/usr/bin/env python
'''
CREATED:2011-11-12 08:23:33 by Brian McFee <bmcfee@cs.ucsd.edu>

Spatial tree demo for matrix data
'''


import numpy
import sys

from spatialtree import spatialtree
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

from spn.structure.Base import Product, Sum, assign_ids, rebuild_scopes_bottom_up
from sklearn.metrics import accuracy_score
from numpy.random.mtrand import RandomState
from spn.algorithms.LearningWrappers import learn_parametric, learn_classifier
import urllib
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.preprocessing import LabelEncoder
numpy.random.seed(42)


def  score(i):
	if i == 'g':
		return 0;
	else:
		return 1;

def one_hot(df,col):
	df = pd.get_dummies([col])
	df.drop()



credit=pd.read_csv("balance_scale.data",delimiter=",") 
credit=credit.drop(columns=credit.columns[0])
print(credit.shape)
from sklearn.model_selection import KFold

kf = KFold(n_splits=10,shuffle=True)

final_theirs = list();
theirs = list()
ours = list();

for train_index, test_index in kf.split(credit):
	X = credit.values[train_index]
	X_test = credit.values[test_index];


	X
	N = X.shape[0]
	D = X.shape[1]
	X_zero = X[X[:,-1]==0]

	#2 1 530101 38.50 66 28 3 3 ? 2 5 4 4 ? ? ? 3 5 45.00 8.40 ? ? 2 2 11300 00000 00000 2
	context = list()
	Gaussian_index = []
	for i in range(0,X.shape[1]):
		context.append(Categorical)





	ds_context = Context(parametric_types=context).add_domains(X)
	print("training normnal spm")
	spn_classification = learn_parametric(numpy.array(X),ds_context,threshold=0.3,min_instances_slice=10)


	ll_original = log_likelihood(spn_classification, X)
	print(numpy.mean(ll_original))
	plot_spn(spn_classification, 'basicspn-original.png')
	ll = log_likelihood(spn_classification, X)
	ll_test = log_likelihood(spn_classification,X_test)
	ll_test_original=ll_test[ll_test>-1000]
	print(numpy.mean(ll_test))
	print(ll_test)



	print('Building tree...')
	T = spatialtree(data=numpy.array(X),ds_context=ds_context,target=X,prob=0.6,height=8,spill=0.75)
	print("Building tree complete")
	T.update_ids()



	spn = T.spn_node_object()
	plot_spn(spn, 'basicspn.png')
	ll = log_likelihood(spn, X)
	ll_test = log_likelihood(spn,X_test)
	ll_test=ll_test[ll_test>-1000]
	ll =ll[ll>-1000]
	print(numpy.mean(ll))
	print(numpy.mean(ll_test))

	theirs.extend(ll_test_original)
	ours.extend(ll_test)
print(numpy.mean(theirs))
print(numpy.mean(ours))




