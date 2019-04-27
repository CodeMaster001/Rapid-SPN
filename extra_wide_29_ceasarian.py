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
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

import pandas as pd

def  score(i):
	if i == 'g':
		return 0;
	else:
		return 1;

kf = KFold(n_splits=10,shuffle=True)
theirs = list()
ours = list();
credit=pd.read_csv("caesarian.csv",delimiter=",") 


print(credit.values.shape)
for train_index, test_index in kf.split(credit):
	X = credit.values[train_index]
	X_test = credit.values[test_index];
	print(X_test)

	
	N = X.shape[0]
	D = X.shape[1]



	context = list()
	left_cols = [Categorical]*D;
	context.extend(left_cols)


	ds_context = Context(parametric_types=context).add_domains(X)

	spn_classification = learn_parametric(X,ds_context,threshold=0.7,rows='tsne')
	plot_spn(spn_classification, 'basicspn-original.png')

	ll_test_original = log_likelihood(spn_classification, X_test)

	l_test_original=ll_test_original[ll_test_original>-1000]

	print('Building tree...')
	T = spatialtree(data=X,ds_context=ds_context,target=X,leaves_size=2,height=2,samples_rp=5,prob=0.25,spill=0.75)
	print("Building tree complete")
	T.update_ids()



	spn = T.spn_node_object()
	plot_spn(spn, 'basicspn.png')
	ll_test = log_likelihood(spn, X_test)
	ll_test=ll_test[ll_test>-1000]
	theirs.extend(ll_test_original)
	ours.extend(ll_test)
print(theirs)
print(ours)
print(numpy.mean(theirs))
print(numpy.mean(ours))


