


import numpy
import sys
import os
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
from sklearn.model_selection import KFold,StratifiedKFold
from spn.gpu.TensorFlow import eval_tf
from spn.structure.Base import *
import time;
import numpy as np, numpy.random
numpy.random.seed(42)
import multiprocessing
import logging
import subprocess
#tf.logging.set_verbosity(tf.logging.INFO)
logging.getLogger().setLevel(logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

# experiment.py train.csv test.csv context.npy instance_slice epochs height prob leaves_sizefor instance_slice in [10,20,40,60]:
train_dataset,labels= fetch_openml(name='blood-transfusion-service-center', version=1,return_X_y=True)
train_dataset_df = pd.DataFrame(train_dataset)
le = LabelEncoder()
labels=le.fit_transform(labels)
kf = StratifiedKFold(n_splits=2,shuffle=True)
theirs = list()
ours = list()
ours_time_list = list()
theirs_time_list = list();
train_set = list()
test_set = list();
counter = 0;
context = list()

#parameters

min_instance_slice=50
epochs=8000
height=1
prob=0.4
leaves_size=18
threshold =0.4
selector_array=[2,3,4]
output_file_name='iris.10.log'
np.save('selector',np.array(selector_array))

opt_args= str(output_file_name) + ' ' + str(min_instance_slice) + ' ' + str(height) +' '+ str(leaves_size) + ' ' +str(threshold)

def bfs(root):
        seen, queue = set([root]), collections.deque([root])
        while queue:
            node = queue.popleft()
            if isinstance(node,Categorical):
                print("--------------")
                node.p=[0.3,0.3,0.4]
                print(node.scope)
                print("---------------")
            if not isinstance(node, Leaf):
                for c in node.children:
                    if c not in seen:
                        seen.add(c)
                        queue.append(c)

def bfsx(root):
        seen, queue = set([root]), collections.deque([root])
        while queue:
            node = queue.popleft()
            if isinstance(node,Categorical):
                print("--------------")
                print(node.p)
                print(node.scope)
                print("---------------")
            if not isinstance(node, Leaf):
                for c in node.children:
                    if c not in seen:
                        seen.add(c)
                        queue.append(c)

for i in range(0,train_dataset_df.shape[1]):
    context.append(Gaussian)
spn_objects = list();
for train_index,test_index in kf.split(train_dataset_df,labels):
    X_train,X_test=train_dataset_df.values[train_index],train_dataset_df.values[test_index]
    Y_train = labels[train_index].reshape(-1,1)
    Y_test = labels[test_index].reshape(-1,1)
    unique = np.unique(Y_train)
    for key in unique:
        X=numpy.nan_to_num(X_train)
        X = X.astype(numpy.float32)
        X = preprocessing.normalize(X, norm='l2')
        X_test = numpy.nan_to_num(X_test)
        X_test = preprocessing.normalize(X_test, norm='l2')
        X = X.astype(numpy.float32)
        X_test =X_test.astype(numpy.float32)
        train_set.append(X)
        test_set.append(X_test)
        np.save('train', X)
        np.save("test",X_test)
        np.save("context",context)
        print(X.shape)
        X=np.append(X,Y_train,axis=1)
        X=X[X[:,-1]==key][:,:-1]
        context = list();
        for i in range(X.shape[1]-1):
            context.append(Gaussian)
        context.append(Categorical)
        ds_context = Context(parametric_types=context).add_domains(X)
        print(X.shape)
        opt_args= str(output_file_name) + ' ' + str(min_instance_slice) + ' ' + str(height) +' '+ str(leaves_size) + ' ' +str(threshold) 
        original = time.time();
        T = SPNRPBuilder(data=numpy.array(X),ds_context=ds_context,bandwidth=0.5,target=X,prob=prob,threshold=threshold,leaves_size=leaves_size,height=height,spill=0.3,selector_array=selector_array,use_optimizer=True,predict_bandwidth=True)
        T= T.build_spn();
        print('updating ids')
        T.update_ids();
        ours_time = time.time()-original;
        spn = T.spn_node;
        spn_objects.append(spn)
        print("Buiding tree complete")
        #file_pi = open(MODEL_DIR+'spnrp_'+str(X.shape[1])+'_'+str(height)+'_'+str(leaves_size)+'.obj', 'wb') 
        #pickle.dump(spn,file_pi)
        ll_test=log_likelihood(spn,X_test)
        print(ll_test)
        #spn=optimize_tf(spn,X,epochs=epochs,optimizer= tf.train.AdamOptimizer(0.00001))
        plot_spn(spn,'spnrp.png')
        S=Sum();
        S.children.append(spn)
        S.weights.append(1.0)
        S=assign_ids(S)
        S.scope.extend(list(range(0,X.shape[1])))
        S=Prune(S)
        plot_spn(S,'spnrpx.png')
        print(X.shape)
        ll_value=log_likelihood(S,X_test)
        print(np.mean(ll_value))
        del spn;
        bfs(S)
        bfsx(S)
        ll_value=log_likelihood(S,X_test)
        print(ll_value)