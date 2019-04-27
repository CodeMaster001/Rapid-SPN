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
from sklearn.model_selection import KFold
from spn.algorithms.Statistics import get_structure_stats

numpy.random.seed(42)
numpy.random.seed(42)


def  score(i):
	if i == 'g':
		return 0;
	else:
		return 1;

def one_hot(df,col):
	df = pd.get_dummies([col])
	df.drop()


credit=pd.read_csv("cropland_pasture.csv",delimiter=",") 
credit = credit.replace(r'^\s+$', numpy.nan, regex=True)

print(credit.shape)
kf = KFold(n_splits=10,shuffle=True)
theirs = list()
ours = list()
credit =credit.drop(credit.columns[1], axis=1)
credit =credit.drop(credit.columns[2], axis=1)
credit =credit.drop(credit.columns[-1], axis=1)
credit =credit.drop(credit.columns[1], axis=1)
print(credit.head())
credit= (credit - credit.mean()) / (credit.max() - credit.min())
credit.values.astype(float)
credit.to_csv('credit.csv')

half_dataset = credit.values[:int(credit.values.shape[0]/2.0),:]
nan_dataset = [np.nan]*half_dataset.shape[0]*half_dataset.shape[1]
nan_dataset =nan_dataset.reshape(half_dataset.shape[0],half_dataset.shape[1])
test_query = np.concatenate((half_dataset, nan_dataset), axis=1)
print(sample_instances(spn, np.array([np.nan, np.nan, np.nan] * 5).reshape(-1, 3), RandomState(123)))

for train_index, test_index in kf.split(credit):
	X = credit.values[train_index]
	X_test = credit.values[test_index];
	print(X.shape)
	
	N = X.shape[0]
	D = X.shape[1]
	X_zero = X[X[:,-1]==0]

	context = list()
	Categorical_index = [0]
	for i in range(0,X.shape[1]):
		context.append(Gaussian)





	ds_context = Context(parametric_types=context).add_domains(X)
	print("training normnal spm")
	spn_classification = learn_parametric(numpy.array(X),ds_context)


	ll_original = log_likelihood(spn_classification, X)
	ll = log_likelihood(spn_classification, X)
	ll_test = log_likelihood(spn_classification,X_test)
	ll_test_original=ll_test[ll_test>-1000]


	print('Building tree...')
	T = spatialtree(data=numpy.array(X),ds_context=ds_context,target=X,prob=0.75,leaves_size=2,height=4,spill=0.25)
	print("Building tree complete")
	T.update_ids()



	spn = T.spn_node_object()
	ll = log_likelihood(spn, X)
	ll_test = log_likelihood(spn,X_test)
	ll_test=ll_test[ll_test>-1000]
	ll =ll[ll>-1000]


	print(numpy.mean(ll_test))
	print(numpy.mean(ll_test_original))
	plot_spn(spn, 'basicspn.png')
	plot_spn(spn_classification, 'basicspn-original.png')
	theirs.extend(ll_test_original)
	ours.extend(ll_test)


print(get_structure_stats(spn))
print(numpy.mean(theirs))
print(numpy.mean(ours))




