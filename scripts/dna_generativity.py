#!/usr/bin/env python 
#2839,180
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
from spn.structure.Base import Product, Sum, assign_ids, rebuild_scopes_bottom_up
from sklearn.metrics import accuracy_score
from numpy.random.mtrand import RandomState
from spn.algorithms.LearningWrappers import learn_parametric, learn_classifier
from spn.algorithms.TransformStructure import Prune,Compress,SPN_Reshape
import urllib
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import fetch_openml
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from spn.structure.Base import *
import time;
import numpy as np, numpy.random
numpy.random.seed(42)
import multiprocessing
import logging
import subprocess
from utils import run_execution_character
from numpy.random.mtrand import RandomState
#tf.logging.set_verbosity(tf.logging.INFO)
logging.getLogger().setLevel(logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
import pickle
import time;
import numpy as np, numpy.random
from spn.algorithms.Sampling import sample_instances
MODEL_DIR='models/dna/'
FILE_NAME_DIR='models/dna/error'

def visualize_spn(X,X_test,context,height=2,prob=0.5,leaves_size=20,bandwidth=0.2,epochs=1000,selector_array=[2,3,4],threshold=1,use_optimizer=True,predict_bandwidth=0.4):
    try:
        print(selector_array)
        ds_context = Context(parametric_types=context).add_domains(X)
        original = time.time();
        T = SPNRPBuilder(data=numpy.array(X),ds_context=ds_context,bandwidth=bandwidth,target=X,prob=prob,threshold=threshold,leaves_size=leaves_size,height=height,spill=0.3,selector_array=selector_array,use_optimizer=use_optimizer,predict_bandwidth=predict_bandwidth)
        T= T.build_spn();
        print('updating ids')
        T.update_ids();
        ours_time = time.time()-original;
        spn = T.spn_node;
        print("Buiding tree complete")
        file_pi = open(MODEL_DIR+'spnrp_'+str(X.shape[1])+'_'+str(height)+'_'+str(leaves_size)+'.obj', 'wb') 
        pickle.dump(spn,file_pi)
        ll_test=log_likelihood(spn,X_test)
        print(ll_test)
        sample = np.asarray([np.nan]*180*100).reshape(100,180)
        print(sample.shape)
        sample = sample_instances(spn,sample,rand_gen=RandomState(42))
        np.save(MODEL_DIR+"numpy_dna",sample)
        #spn=optimize_tf(spn,X,epochs=epochs,optimizer= tf.train.AdamOptimizer(0.00001))
        #plot_spn(spn,'spnrp.png')
        print(sample)
        del spn;
        return np.mean(ll_test),ours_time
    except:
        f=open(FILE_NAME_DIR+'error.log','a')
        f.write(traceback.format_exc()+"\n")
        traceback.print_exc()
        f.flush()
        f.close()
        return 0,0

# experiment.py train.csv test.csv context.npy instance_slice epochs height prob leaves_size

train_dataset,labels= fetch_openml(name='dna', version=1,return_X_y=True)
le = preprocessing.LabelEncoder()
train_dataset = train_dataset[train_dataset.columns].apply(le.fit_transform)
kf = KFold(n_splits=10,shuffle=True)
theirs = list()
ours = list()
ours_time_list = list()
theirs_time_list = list();
train_set = list()
test_set = list();
counter = 0;
context = list()

#parameters
output_file_name='dna.10.csv'
min_instance_slice=40
epochs=8000
height=22
prob=0.4
leaves_size=6
threshold =0.4


train_sample ,test_sample = train_test_split(train_dataset,test_size=0.25)
#################################################Build-Context########################################
context = [] 
print(train_sample.shape)
for i in range(0,train_sample.shape[1]):
        context.append(Categorical)
print(test_sample.shape)
print(len(context))
ds_context = Context(parametric_types=context).add_domains(np.asarray(train_sample))
spnrp_mean,spnrp_time = visualize_spn(X=np.asarray(train_sample),X_test=np.asarray(test_sample),context=context,height=height,leaves_size=leaves_size,threshold=threshold)


