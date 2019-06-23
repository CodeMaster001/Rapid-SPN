#!/usr/bin/env python
'''
CREATED:2011-11-12 08:23:33 by Brian McFee <bmcfee@cs.ucsd.edu>

Spatial tree demo for matrix data
'''


import numpy
import sys
import os

import random
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
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np, numpy.random
numpy.random.seed(42)
import logging
logger = logging.getLogger('spnrp')
# create file handler which logs even debug messages
logging.basicConfig(filename='spnrpp.log',level=logging.DEBUG)

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
from typing import Union, Tuple, List

from spn.algorithms.TransformStructure import Copy
from spn.structure.Base import Product, Sum, eval_spn_bottom_up, Node
from spn.structure.leaves.histogram.Histograms import Histogram
from spn.structure.leaves.histogram.Inference import histogram_likelihood
from spn.structure.leaves.parametric.Parametric import Gaussian



def log_sum_to_tf_graph(node, children, data_placeholder=None, variable_dict=None, log_space=True, dtype=np.float32):
    assert log_space
    with tf.variable_scope("%s_%s" % (node.__class__.__name__, node.id)):
        softmaxInverse = np.log(node.weights / np.max(node.weights)+0.00000001).astype(dtype)
        tfweights = tf.nn.softmax(tf.get_variable("weights", initializer=tf.constant(softmaxInverse)))
        variable_dict[node] = tfweights
        childrenprob = tf.stack(children, axis=1)
        return tf.reduce_logsumexp(childrenprob + tf.log(tfweights), axis=1)


def tf_graph_to_sum(node, tfvar):
    node.weights = tfvar.tolist()


def log_prod_to_tf_graph(node, children, data_placeholder=None, variable_dict=None, log_space=True, dtype=np.float32):
    assert log_space
    with tf.variable_scope("%s_%s" % (node.__class__.__name__, node.id)):
        return tf.add_n(children)


def histogram_to_tf_graph(node, data_placeholder=None, log_space=True, variable_dict=None, dtype=np.float32):
    with tf.variable_scope("%s_%s" % (node.__class__.__name__, node.id)):
        inps = np.arange(int(max(node.breaks))).reshape((-1, 1))
        tmpscope = node.scope[0]
        node.scope[0] = 0
        hll = histogram_likelihood(node, inps)
        node.scope[0] = tmpscope
        if log_space:
            hll = np.log(hll)

        lls = tf.constant(hll.astype(dtype))

        col = data_placeholder[:, node.scope[0]]

        return tf.squeeze(tf.gather(lls, col))


_node_log_tf_graph = {Sum: log_sum_to_tf_graph, Product: log_prod_to_tf_graph, Histogram: histogram_to_tf_graph}


def add_node_to_tf_graph(node_type, lambda_func):
    _node_log_tf_graph[node_type] = lambda_func


_tf_graph_to_node = {Sum: tf_graph_to_sum}


def add_tf_graph_to_node(node_type, lambda_func):
    _tf_graph_to_node[node_type] = lambda_func


def spn_to_tf_graph(node, data, batch_size=None, node_tf_graph=_node_log_tf_graph, log_space=True, dtype=None):
    tf.reset_default_graph()
    if not dtype:
        dtype = data.dtype
    # data is a placeholder, with shape same as numpy data
    data_placeholder = tf.placeholder(data.dtype, (batch_size, data.shape[1]))
    variable_dict = {}
    tf_graph = eval_spn_bottom_up(
        node,
        node_tf_graph,
        data_placeholder=data_placeholder,
        log_space=log_space,
        variable_dict=variable_dict,
        dtype=dtype,
    )
    return tf_graph, data_placeholder, variable_dict


def tf_graph_to_spn(variable_dict, tf_graph_to_node=_tf_graph_to_node):
    tensors = []

    for n, tfvars in variable_dict.items():
        tensors.append(tfvars)

    variable_list = tf.get_default_session().run(tensors)

    for i, (n, tfvars) in enumerate(variable_dict.items()):
        tf_graph_to_node[type(n)](n, variable_list[i])


def likelihood_loss(tf_graph):
    # minimize negative log likelihood
    return -tf.reduce_sum(tf_graph)



def optimize_tf(
    spn: Node,
    data: np.ndarray,
    epochs=1000,
    batch_size: int = None,
    optimizer: tf.train.Optimizer = None,
    return_loss=False,
) -> Union[Tuple[Node, List[float]], Node]:
    """
    Optimize weights of an SPN with a tensorflow stochastic gradient descent optimizer, maximizing the likelihood
    function.
    :param spn: SPN which is to be optimized
    :param data: Input data
    :param epochs: Number of epochs
    :param batch_size: Size of each minibatch for SGD
    :param optimizer: Optimizer procedure
    :param return_loss: Whether to also return the list of losses for each epoch or not
    :return: If `return_loss` is true, a copy of the optimized SPN and the list of the losses for each epoch is
    returned, else only a copy of the optimized SPN is returned
    """
    # Make sure, that the passed SPN is not modified
    spn_copy = Copy(spn)

    # Compile the SPN to a static tensorflow graph
    tf_graph, data_placeholder, variable_dict = spn_to_tf_graph(spn_copy, data, batch_size)

    # Optimize the tensorflow graph
    loss_list = optimize_tf_graph(
        tf_graph, variable_dict, data_placeholder, data, epochs=epochs, batch_size=batch_size, optimizer=optimizer
    )

    # Return loss as well if flag is set
    if return_loss:
        return spn_copy, loss_list

    return spn_copy


def optimize_tf_graph(
    tf_graph, variable_dict, data_placeholder, data, epochs=1000, batch_size=None, optimizer=None
) -> List[float]:
    if optimizer is None:
        optimizer = tf.train.GradientDescentOptimizer(0.001)
    loss = -tf.reduce_sum(tf_graph)
    original_optimizer = tf.train.AdamOptimizer(learning_rate=0.00000001)
    optimizer = tf.contrib.estimator.clip_gradients_by_norm(original_optimizer, clip_norm=5.0)
    opt_op = optimizer.minimize(loss)

    # Collect loss
    loss_list = [0]
    config = tf.ConfigProto(
        device_count = {'GPU': 0})
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        if not batch_size:
            batch_size = data.shape[0]
        batches_per_epoch = data.shape[0] // batch_size
        old_loss = 0;
        # Iterate over epochs
        while  True:
  

            # Collect loss over batches for one epoch
            epoch_loss = 0.0

            # Iterate over batches
            for j in range(batches_per_epoch):
                data_batch = data[j * batch_size : (j + 1) * batch_size, :]
         
                _, batch_loss = sess.run([opt_op, loss], feed_dict={data_placeholder: data_batch})
           
                epoch_loss += batch_loss
              
           
            # Build mean
            epoch_loss /= data.shape[0]


            print("Epoch: %s, Loss: %s", i, epoch_loss)
            loss_list.append(epoch_loss)
            old_loss = np.abs(loss_list[-1]) - np.abs(loss_list[-2])
            print(old_loss)
            if np.abs(old_loss) < 0.0002:
         	   break;

        tf_graph_to_spn(variable_dict)

    return loss_list




def bfs(root, func):
    seen, queue = set([root]), collections.deque([root])
    while queue:
        node = queue.popleft()
        func(node)
        if not isinstance(node, Leaf):
            for c in node.children:
                if c not in seen:
                    seen.add(c)
                    queue.append(c)

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






credit = pd.read_csv('2012Q1-capitalbikeshare-tripdata.csv', delimiter=',')
credit = credit.drop('Bike number',axis=1)
print(credit.columns)
credit = credit.drop('Start date',axis=1)
credit= credit.drop('End date',axis=1)
credit_gaussian = credit.iloc[:,[0]]
credit_gaussian= numpy.nan_to_num(credit_gaussian.values)
print(credit_gaussian[:10,:])
credit_gaussian = preprocessing.normalize(credit_gaussian,norm='l2')
columns =[0]
column_names = list(credit.columns)
for col in range(0,len(credit.columns)):
    if col in columns:
        del credit[column_names[col]]
#credit = credit.drop(['manufacturer'],axis=1)
credit_categorical = credit;
print(credit_categorical.head(10))
credit_categorical = credit_categorical.fillna(0)
le = preprocessing.LabelEncoder()
# 2/3. FIT AND TRANSFORM
credit_categorical = credit_categorical.astype(str)
credit_categorical = credit_categorical.apply(le.fit_transform)
credit = np.concatenate((credit_gaussian, credit_categorical), axis=1)
print(credit[10,:])
kf = KFold(n_splits=10,shuffle=True)
theirs = list()
ours = list()
ours_time_list = list()
theirs_time_list = list();
ours_time_tf = list()
theirs_time_tf = list();
column_index = [0]
for train_index, test_index in kf.split(credit):
    X = credit[train_index,:]
    print(X.shape)
    X_test = credit[test_index];	
    X = X.astype(numpy.float32)
    X_test =X_test.astype(numpy.float32)
    context = list()
    for i in range(0,X.shape[1]):
        if i in column_index:
            context.append(Gaussian)
        else:
            context.append(Categorical)


    ds_context = Context(parametric_types=context).add_domains(X)
    print("training normnal spm")

    original = time.time()
    spn_classification =  learn_parametric(numpy.array(X),ds_context,min_instances_slice=5000000,threshold=0.6)

    #spn_classification = optimize_tf(spn_classification,X,epochs=1000,optimizer= tf.train.AdamOptimizer(0.001)) 
    #tf.train.AdamOptimizer(1e-4))

    theirs_time = time.time()-original


    #ll_test = eval_tf(spn_classification, X_test)
    #print(ll_test)
    ll_test = log_likelihood(spn_classification,X_test)
    theirs_time_tf = time.time() -original

    ll_test_original=ll_test[ll_test>-1000]


    logging.info('Building tree...')
    original = time.time();
    T = SPNRPBuilder(data=numpy.array(X),ds_context=ds_context,target=X,prob=0.7,leaves_size=2,height=4,spill=0.3)
    logging.info("Building tree complete")

    T= T.build_spn();
    T.update_ids();
    from spn.io.Text import spn_to_str_equation
    spn = T.spn_node;
    ours_time = time.time()-original
    ours_time_list.append(ours_time)
    #bfs(spn,print_prob)
    ll_test = log_likelihood(spn, X_test)
    spn=optimize_tf(spn,X,epochs=60000,optimizer= tf.train.AdamOptimizer(0.001))
    #ll_test = eval_tf(spn,X)
    ours_time_tf = time.time()-original
    ll_test=ll_test[ll_test>-1000]
    print('completed')
    logging.info("--ll--")
    logging.info(numpy.mean(ll_test_original))
    logging.info(numpy.mean(ll_test))
    logging.info(theirs_time)
    logging.info(ours_time)
    logging.info(theirs_time_tf)
    logging.info(ours_time_tf)
    theirs.append(numpy.mean(ll_test_original))
    ours.append(numpy.mean(ll_test))
    theirs_time_list.append(theirs_time)
    sys.exit(-1)
    #plot_spn(spn_classification, 'basicspn-original.png')
#plot_spn(spn, 'basicspn.png')
logging.info(theirs)
logging.info(ours)
logging.info(original)
logging.info('---Time---')
logging.info(numpy.mean(theirs_time_list))
logging.info(numpy.var(theirs_time_list))
logging.info(numpy.mean(ours_time_list))
logging.info(numpy.var(ours_time_list))
logging.info('---ll---')
logging.info(numpy.mean(theirs))
logging.info(numpy.var(theirs))

logging.info(numpy.mean(ours))
logging.info(numpy.var(ours))





