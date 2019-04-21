#!/usr/bin/env python
'''
CREATED:2011-11-12 08:23:33 by Brian McFee <bmcfee@cs.ucsd.edu>

Spatial tree demo for matrix data
'''


import numpy
import sys
sys.path.append("fashion-mnist/utils/")

from spatialtree import spatialtree
from spn.structure.Base import Context
from spn.io.Graphics import plot_spn
from spn.structure.leaves.parametric.Parametric import Categorical, Gaussian
from spn.algorithms.Inference import log_likelihood
from spn.algorithms.splitting.RDC import get_split_cols_RDC_py, get_split_rows_RDC_py
from sklearn.datasets import load_iris,load_digits,fetch_california_housing
from sklearn.model_selection import train_test_split
import tensorflow as tf
import mnist_reader
X_train, y_train = mnist_reader.load_mnist('fashion-mnist/data/fashion', kind='train')



# First, create a random data matrix
X = numpy.concatenate((X_train, y_train.reshape(-1,1)),axis=1)
N = X.shape[0]
D = X.shape[1]



# Apply a random projection so the data's not totally boring

#X = numpy.dot(X, P)

# Construct a tree.  By default, we get a KD-spill-tree with height
# determined automatically, and spill = 25%
context = [Gaussian]*(D-1);
context.append(Categorical)

ds_context = Context(parametric_types=context).add_domains(X)


print('Building tree...')
T = spatialtree(X,ds_context=ds_context,leaves_size=2000)
print("Building tree complete")
T.update_ids()
spn = T.spn_node_object()


plot_spn(spn, 'basicspn.png')
ll = log_likelihood(spn, X_test)
print(numpy.mean(ll))
