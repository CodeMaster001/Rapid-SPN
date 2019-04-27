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
from sklearn.metrics import accuracy_score
from numpy.random.mtrand import RandomState
from spn.algorithms.TransformStructure import Prune,Compress
from spn.algorithms.LearningWrappers import learn_parametric, learn_classifier


from spn.structure.Base import Product, Sum, assign_ids, rebuild_scopes_bottom_up
from spn.algorithms.LearningWrappers import learn_parametric, learn_classifier
import tensorflow as tf
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.preprocessing import LabelEncoder
mnist = load_breast_cancer()
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
credit=pd.read_csv("OVA_Ovary.csv",delimiter=",") 
kf = KFold(n_splits=10,shuffle=True)
credit = credit.drop(credit.columns[-1], axis=1)
print(credit.shape)
theirs = list()
ours = list();
for train_index, test_index in kf.split(credit):
	X = credit.values[train_index]
	X_test = credit.values[test_index];
	N = X.shape[0]
	D = X.shape[1]

	# First, create a random data matrix



	# Apply a random projection so the data's not totally boring

	#X = numpy.dot(X, P)

	# Construct a tree.  By default, we get a KD-spill-tree with height
	# determined automatically, and spill = 25%
	context = [Gaussian]*D

	ds_context = Context(parametric_types=context).add_domains(X)


	spn_classification = learn_parametric(X,ds_context)


	ll_test_original= log_likelihood(spn_classification, X_test)

	#ll_test = log_likelihood(spn_original, X_test)






	ds_context = Context(parametric_types=context).add_domains(X)
	print('Building tree...')
	T = spatialtree(data=X,ds_context=ds_context,leaves_size=20,target=X.shape[1]-1,rule='rp',prob=0.2,height=3,spill=0.76)
	print("Building tree complete")
	T.update_ids()
	spn = T.spn_node_object()


	ll_test= log_likelihood(spn, X_test)
	ll_test=ll_test[ll_test>-1000]
	ll_test_original =ll_test_original[ll_test_original>-1000]
	print(numpy.mean(ll_test_original))
	print(numpy.mean(ll_test))
	break;
	theirs.extend(ll_test_original)
	ours.extend(ll_test)
print(numpy.mean(theirs))
print(numpy.mean(ours))




#print(numpy.mean(ll_test))