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
import numpy as np, numpy.random
numpy.random.seed(42)
import logging


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
    original_optimizer = tf.train.AdamOptimizer(learning_rate=0.00001)
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
        for i in range(0,epochs):
  

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
            if old_loss<0.02:
                break;
            print(old_loss)

        tf_graph_to_spn(variable_dict)

    return loss_list



#tf.logging.set_verbosity(tf.logging.INFO)
logging.getLogger().setLevel(logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

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





credit = fetch_openml(name='cars', version=2,return_X_y=True)[0]
credit = pd.DataFrame(credit)
credit = credit.fillna(0)
kf = KFold(n_splits=40,shuffle=True)
theirs = list()
ours = list()
ours_time_list = list()
theirs_time_list = list();
counter =0;
for train_index, test_index in kf.split(credit):

    print(train_index)
    X = credit.values[train_index,:]
    X=numpy.nan_to_num(X)
    #X = preprocessing.normalize(X, norm='l2')
    X_test = credit.values[test_index];	
    X_test = numpy.nan_to_num(X_test)
    #X_test = preprocessing.normalize(X_test, norm='l2')
    X = X.astype(numpy.float32)
    X_test =X_test.astype(numpy.float32)
    context = list()
    for i in range(0,X.shape[1]):
        context.append(Gaussian)

	



    ds_context = Context(parametric_types=context).add_domains(X)
    print("training normnal spm")

    theirs_time = time.time()
    spn_classification =  learn_parametric(numpy.array(X),ds_context)
    spn_classification = optimize_tf(spn_classification,X,epochs=5000,optimizer= tf.train.AdamOptimizer(0.001)) 
    #tf.train.AdamOptimizer(1e-4))

    theirs_time = time.time()-theirs_time


    ll_test = eval_tf(spn_classification, X_test)
    #print(ll_test)
    #ll_test = log_likelihood(spn_classification,X_test)
    ll_test_original=ll_test[ll_test>-1000]




    print('Building tree...')
    original = time.time();
   
    T = SPNRPBuilder(data=numpy.array(X),ds_context=ds_context,target=X,prob=0.4,height=4,leaves_size=100)
    print("Building tree complete")

    T= T.build_spn();
    T.update_ids();
    from spn.io.Text import spn_to_str_equation
    spn = T.spn_node;
    ours_time = time.time()-original;
    ours_time_list.append(ours_time)
    bfs(spn,print_prob)
    ll = log_likelihood(spn, X)
    spn=optimize_tf(spn,X,epochs=60000,optimizer= tf.train.AdamOptimizer(0.001))
    ll_test = eval_tf(spn,X)
    ll_test=ll_test[ll_test>-1000]
    print("--ll--")
    print("tt:"+str(counter)+":"+str(numpy.mean(ours_time_list)))
    print("tt:"+str(counter)+":"+str(numpy.mean(theirs_time_list)))
    print("ll:"+str(counter)+":"+str(numpy.mean(ll_test_original)))
    print("ll:"+str(counter)+":"+str(numpy.mean(ll_test)))
    print("---ended---")
    counter = counter + 1
    del spn
    del spn_classification
    del T
    
    theirs.append(numpy.mean(ll_test_original))
    ours.append(numpy.mean(ll_test))
    theirs_time_list.append(theirs_time)
 

#plot_spn(spn_classification, 'basicspn-original.png')
#plot_spn(spn, 'basicspn.png')
print(theirs)
print(ours)
print(original)
print('---Time---')
print(numpy.mean(theirs_time_list))
print(numpy.var(theirs_time_list))
print(numpy.mean(ours_time_list))
print(numpy.var(ours_time_list))
print('---ll---')
print(numpy.mean(theirs))
print(numpy.var(theirs))

print(numpy.mean(ours))
print(numpy.var(ours))
theirs = np.array(theirs)
ours_time_list = np.array(ours_time_list)
theirs_time_list = np.array(theirs_time_list)
result = np.vstack((ours,theirs))
np.savetxt('cars.outll',result,delimiter=',')
result = np.vstack((ours_time_list,theirs_time_list))
np.savetxt('cars.outtt',result,delimiter=',')
print(numpy.mean(theirs))


