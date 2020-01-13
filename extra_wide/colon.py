#!/usr/bin/env python
'''
CREATED:2011-11-12 08:23:33 by Brian McFee <bmcfee@cs.ucsd.edu>

Spatial tree demo for matrix data
'''


import numpy
import sys
import os
os.environ['NUMEXPR_MAX_THREADS'] = '16'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn import preprocessing
from sklearn.datasets import load_svmlight_file
from spatialtree import *;
from sklearn.model_selection import train_test_split
from libsvm.svmutil import *
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
import multiprocessing
import logging
import subprocess
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
    tf.reset_default_graph() 
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
    optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, clip_norm=5.0)
    opt_op = optimizer.minimize(loss)
    i = 0;
    # Collect loss
    loss_list = [0]
    config =tf.ConfigProto(log_device_placement=True,);
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        if not batch_size:
            batch_size = data.shape[0]
        batches_per_epoch = data.shape[0] // batch_size
        old_loss = 0;
        # Iterate over epochs
        while  i<epochs:
            i = i+1;
  

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
            loss_list.append(old_loss)

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


def clean_data(x):
	try:
		return str(x).split(':')[-1]
	except:
		print(str(x))


def libsvm_preprocess(file_name,row_size=62,colon_size=200):
	dataset_label, dataset_feature = svm_read_problem('../dataset/colon.csv')
	X=list()
	Y=list()
	for i in range(0,row_size):
		temp = list();
		for j in range(0,colon_size):
			temp.append(dataset_feature[i][j+1])
		X.append(temp)
		Y.append(dataset_label[i])
	return np.array(X),np.array(Y)
 

#print(credit.head())

def spnrp_train(X,X_test):
    context = list()
    for i in range(0,X.shape[1]):
        context.append(Gaussian)


    ds_context = Context(parametric_types=context).add_domains(X)
    original = time.time();
    T = SPNRPBuilder(data=numpy.array(X),ds_context=ds_context,target=X,prob=0.5,leaves_size=3,height=3,spill=0.3)
    print("Buiding tree complete")
    print("done")
    T= T.build_spn();
    #T.update_ids();
    ours_time = time.time()-original;
    spn = T.spn_node;
    plot_spn(spn, 'basicspn-original.png')
    spn=optimize_tf(spn,X,epochs=1000)
    ll_test = eval_tf(spn,X_test)
    tf.reset_default_graph()
    del spn;
    return np.mean(ll_test),ours_time

def learnspn_train(X,X_test):
	context = list()
	for i in range(0,X.shape[1]):
	    context.append(Gaussian)




	ds_context = Context(parametric_types=context).add_domains(X)
	theirs_time = time.time()
	spn_classification =  learn_parametric(numpy.array(X),ds_context,min_instances_slice=5)
	theirs_time = time.time()-theirs_time
	spn_classification = optimize_tf(spn_classification,X,epochs=1000,optimizer= tf.train.AdamOptimizer(0.0001)) 
	    #tf.train.AdamOptimizer(1e-4))


	ll_test = eval_tf(spn_classification,X_test)
	#print(ll_test)
	#ll_test = log_likelihood(spn_classification,X_test)
	ll_test_original=ll_test
	tf.reset_default_graph()
	del spn_classification
	return  np.mean(ll_test),theirs_time

# experiment.py train.csv test.csv context.npy instance_slice epochs height prob leaves_size

dataset_feature,dataset_label = libsvm_preprocess('../dataset/colon.csv')
print(dataset_feature)

kf = KFold(n_splits=5,shuffle=True)
theirs = list()
ours = list()
ours_time_list = list()
theirs_time_list = list();
train_set = list()
test_set = list();
counter = 0;
context = list()
for i in range(0,dataset_feature.shape[1]):
    context.append(Gaussian)
for train_index,test_index in kf.split(dataset_feature):
    X_train,X_test=dataset_feature[train_index],dataset_feature[test_index]
    X=numpy.nan_to_num(X_train)
    X = X.astype(numpy.float32)
    X = preprocessing.normalize(X, norm='l2') 
    X_test = numpy.nan_to_num(X_test)
    X_test = preprocessing.normalize(X_test, norm='l2')
    X = X.astype(numpy.float32)
    X_test =X_test.astype(numpy.float32)
    train_set.append(X)
    test_set.append(X_test)
    np.savetxt('train.csv', X, delimiter=',')
    np.savetxt("test.csv",X_test,delimiter=',')
    np.save("context",context)
    P=subprocess.Popen(['./experiment.py train.csv test.csv context.npy 5 20 3 0.4 20'],shell=True)
    P.communicate()
    P.wait();
    P.terminate()
    sys.exit(0)
print("process completed")
