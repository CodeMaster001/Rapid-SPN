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


def  score(i):
	if i == 'g':
		return 0;
	else:
		return 1;

def one_hot(df,col):
	df = pd.get_dummies([col])
	df.drop()



credit=pd.read_csv("isolet5.data",delimiter=",") 
credit = credit.apply(LabelEncoder().fit_transform)
credit_x = credit.values[:,:-1]
print(credit.shape)
credit_y = credit.values[:,-1]
X_train, X_test, y_train, y_test = train_test_split(credit.values[:,:-1],credit.values[:,-1],test_size=0.1) 


# First, create a random data matrix
X = numpy.concatenate((X_train, y_train.reshape(-1,1)),axis=1)
X_test = numpy.concatenate((X_test, y_test.reshape(-1,1)),axis=1)


X
N = X.shape[0]
D = X.shape[1]
X_zero = X[X[:,-1]==0]


context = list()
left_cols = [Gaussian]*(D-1);
context.extend(left_cols)
context.append(Categorical)


ds_context = Context(parametric_types=context).add_domains(X)
print("training normnal spm")
spn_classification = learn_parametric(numpy.array(X),ds_context)


ll_original = log_likelihood(spn_classification, X)
print(numpy.mean(ll_original))
plot_spn(spn_classification, 'basicspn-original.png')
ll = log_likelihood(spn_classification, X)
ll_test = log_likelihood(spn_classification,X_test)


print(numpy.mean(ll))
print(numpy.mean(ll_test))

print('Building tree...')
T = spatialtree(data=numpy.array(X),ds_context=ds_context,target=X,prob=0.7,leaves_size=20)
print("Building tree complete")
T.update_ids()



spn = T.spn_node_object()
plot_spn(spn, 'basicspn.png')
ll = log_likelihood(spn, X)
ll_test = log_likelihood(spn,X_test)
ll_test=ll_test[ll_test>-1000]
ll_original=ll_original[ll_original>-1000]

print(numpy.mean(ll))
print(numpy.mean(ll_test))




