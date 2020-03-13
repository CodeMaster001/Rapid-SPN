#!/usr/bin/env python
'''

Spatial tree demo for matrix data
# experiment.py train.csv test.csv context.npy instance_slice epochs height prob leaves_size
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
import pickle
import time;
import numpy as np, numpy.random
import sys;

from pathlib import Path
FILE_NAME_DIR="results/"
Path("results").mkdir(parents=True, exist_ok=True)
import multiprocessing
import logging
import traceback

"""
def optimize_tf(
    spn: Node,
    data: np.ndarray,
    epochs=1000,
    batch_size: int = None,
    optimizer: tf.train.Optimizer = None,
    return_loss=False,
) -> Union[Tuple[Node, List[float]], Node]:
    
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
    counter = 0;
    config =tf.ConfigProto(log_device_placement=True,);
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        if not batch_size:
            batch_size = data.shape[0]
        batches_per_epoch = data.shape[0] // batch_size
        old_loss = 0;
        # Iterate over epochs
        while i<epochs:
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
        
            if np.abs(old_loss) < 0.0005:
                counter = counter + 1;
                if counter>10:
                    break;
            else:
                counter = 0;
        tf_graph_to_spn(variable_dict)

    return loss_list

"""

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

 

#print(credit.head())

def spnrp_train(X,X_test,context,height=2,prob=0.5,leaves_size=20,bandwidth=0.2,epochs=1000,selector_array=[2,3,4],threshold=1,use_optimizer=True,predict_bandwidth=0.4):
    try:
        print(selector_array)

        ds_context = Context(parametric_types=context).add_domains(X)
        original = time.time();
        T = SPNRPBuilder(data=numpy.array(X),ds_context=ds_context,bandwidth=bandwidth,target=X,prob=prob,threshold=threshold,leaves_size=leaves_size,height=height,spill=0.3,selector_array=selector_array,use_optimizer=use_optimizer,predict_bandwidth=predict_bandwidth)
        T= T.build_spn();
        T.update_ids();
        ours_time = time.time()-original;
        spn = T.spn_node;
        print("Buiding tree complete")
        ll_test=log_likelihood(spn,X_test)
        print(ll_test)
        #spn=optimize_tf(spn,X,epochs=epochs,optimizer= tf.train.AdamOptimizer(0.00001))
        #plot_spn(spn,'spnrp.png')
        #ll_test = eval_tf(spn,X_test)
        tf.reset_default_graph();
        del spn;
        return np.mean(ll_test),ours_time
    except:
        f=open(FILE_NAME_DIR+'error.log','a')
        f.write(traceback.format_exc()+"\n")
        traceback.print_exc()
        f.flush()
        f.close()
        return 0,0


def learnspn_train(X,X_test,context,min_instances_slice,threshold=0.4):
    
    try:



        ds_context = Context(parametric_types=context).add_domains(X)
        theirs_time = time.time()
        spn_classification =  learn_parametric(numpy.array(X),ds_context,min_instances_slice=min_instances_slice,threshold=threshold)
        theirs_time = time.time()-theirs_time
        #spn_classification = optimize_tf(spn_classification,X,epochs=epochs,optimizer= tf.train.AdamOptimizer(0.0001)) 
            #tf.train.AdamOptimizer(1e-4))


        #ll_test = eval_tf(spn_classification,X_test)
        #print(ll_test)
        ll_test = log_likelihood(spn_classification,X_test)
        #plot_spn(spn_classification,'learnspn.png')
    
        del spn_classification
        return  np.mean(ll_test),theirs_time
    except:
        f=open(FILE_NAME_DIR+'error.log','a')
        f.write(str(sys.argv)+"\n")
        f.write(traceback.format_exc())
        f.flush()
        f.close();
        sys.exit(-1)

# train.npy test.npy context.npy train_context_filename output_file_name test_file_name min_instance_slice epochs height prob leaves_size 
print(sys.argv)
train_file_name=sys.argv[1]
test_file_name=sys.argv[2]
context = np.load(sys.argv[3],allow_pickle=True)
file_name=str(sys.argv[4])
min_instances_slice = int(sys.argv[5])
height=int(sys.argv[6])
leaves_size=float(sys.argv[7])
threshold = float(sys.argv[8])
spn_mean=0
spn_time=0

X=np.load(train_file_name)
X_test=np.load(test_file_name)
print('Test')
print(X_test.shape)
print('Train')
print(X.shape)
spn_mean=0
spn_time=0
assert X.shape[1]==X_test.shape[1]
spnrp_mean=0
spnrp_time=0
spn_mean,spn_time = learnspn_train(X,X_test,context,min_instances_slice,threshold)
spnrp_mean,spnrp_time = spnrp_train(X=X,X_test=X_test,context=context,height=height,leaves_size=leaves_size,threshold=threshold)
f=open(FILE_NAME_DIR+file_name,'a')
f.write(str(sys.argv)+"\n")
print(spnrp_mean)
print(FILE_NAME_DIR+file_name)
temp=str(spn_mean)+","+str(spnrp_mean)+","+str(spn_time)+","+str(spnrp_time)+","+str(min_instances_slice)+"\n"
print(temp)
f.write(temp)
f.flush()
f.close()



