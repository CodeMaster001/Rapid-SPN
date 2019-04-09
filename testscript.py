#!/usr/bin/env python
'''
CREATED:2011-11-12 08:23:33 by Brian McFee <bmcfee@cs.ucsd.edu>

Spatial tree demo for matrix data
'''


import numpy
import sys

from spatialtree import spatialtree,Classifier
from spn.structure.Base import Context
from spn.io.Graphics import plot_spn
from spn.algorithms.Sampling import sample_instances
from spn.structure.leaves.parametric.Parametric import Categorical, Gaussian
from spn.algorithms.Inference import log_likelihood
from spn.algorithms.splitting.RDC import get_split_cols_RDC_py, get_split_rows_RDC_py
from sklearn.datasets import load_iris,load_digits,fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score
from numpy.random.mtrand import RandomState

mnist = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split (mnist.data, mnist.target, test_size=0.33, random_state=42)
X_test_prediction = numpy.array([numpy.nan]*y_test.shape[0]).reshape(-1,1)
X_test_prediction =numpy.concatenate((X_test, X_test_prediction),axis=1)


# First, create a random data matrix
print(mnist.target)
X = numpy.concatenate((X_train, y_train.reshape(-1,1)),axis=1)
N = X.shape[0]
D = X.shape[1]

X_test = numpy.concatenate((X_test, y_test.reshape(-1,1)),axis=1)



# Apply a random projection so the data's not totally boring

#X = numpy.dot(X, P)

# Construct a tree.  By default, we get a KD-spill-tree with height
# determined automatically, and spill = 25%
context = [Gaussian]*(D-1);
context.append(Categorical)
ds_context = Context(parametric_types=context).add_domains(X)


print('Building tree...')
print(numpy.unique(X_test[:,X_test.shape[1]-1]))
T = Classifier();
root= T.build_classifier(X,ds_context=ds_context,leaves_size=5,target=X_test.shape[1]-1)



plot_spn(root, 'basicspn.png')
ll = log_likelihood(root, X_test)

result = sample_instances(root, X_test_prediction, RandomState(123))
pred = result[:,-1]
score = accuracy_score(pred, y_test)
print(score)
print(numpy.mean(ll))
