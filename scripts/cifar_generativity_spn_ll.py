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
from scipy.stats._result_classes import FitResult
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
from sklearn.decomposition import PCA
def train_spn_pca(X,X_test,context,height=2,prob=0.5,leaves_size=20,min_instance_slice=40,bandwidth=0.2,epochs=1000,selector_array=[2,3,4],threshold=1,use_optimizer=True,predict_bandwidth=0.4):
    try:
        print(selector_array)
        print(X.shape)
        print(X_test.shape)
        ds_context = Context(parametric_types=context).add_domains(X)
        original = time.time();
        spn =  learn_parametric(numpy.array(X),ds_context,min_instances_slice=min_instance_slice,threshold=threshold)
        ours_time = time.time()-original
        print("Buiding tree complete")
        ll_test=log_likelihood(spn,X_test)
        
        print(ll_test)
        print(ll_test)
        sample = np.asarray([np.nan]*X.shape[1]*X.shape[0]).reshape(X.shape[0],X.shape[1]) #change the shape to no of components
        print(sample.shape)
        sample = sample_instances(spn,sample,rand_gen=RandomState(42))
        #spn=optimize_tf(spn,X,epochs=epochs,optimizer= tf.train.AdamOptimizer(0.00001))
        #plot_spn(spn,'spnrp.png')
        print(sample.shape)
        del spn;
        return np.mean(ll_test),ours_time,sample
    except:
        f=open(FILE_NAME_DIR+'error.log','a')
        f.write(traceback.format_exc()+"\n")
        traceback.print_exc()
        f.flush()
        f.close()
        return 0,0

# experiment.py train.csv test.csv context.npy instance_slice epochs height prob leaves_size

train_dataset,labels= fetch_openml(name='CIFAR_10', version=1,return_X_y=True)
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
height=1
prob=0.4
leaves_size=30
threshold =0.4

pca = PCA(n_components=100)

train_sample ,test_sample = train_test_split(train_dataset,test_size=0.25)
#################################################Build-Context########################################
context = [] 
pca_component_train = pca.fit(train_sample)
pca_component_test = pca.fit(test_sample)
train_sample = pca_component_train.fit_transform(train_sample)
test_sample = pca_component_test.transform(test_sample)
print(test_sample.shape)
print(train_sample.shape)
for i in range(0,train_sample.shape[1]):
        context.append(Gaussian)
print(test_sample.shape)
print(len(context))
ds_context = Context(parametric_types=context).add_domains(np.asarray(train_sample))
spnrp_mean,spnrp_time,sample = train_spn_pca(X=np.asarray(train_sample),X_test=np.asarray(test_sample),context=context,height=height,leaves_size=leaves_size,threshold=threshold)
print("Train Model completed..Now Calculating inverse transform")
print(sample.shape)
model_sample = pca_component_test.inverse_transform(sample)
#test_dataset = pca_component_test.inverse_transform(test_sample)
np.save(MODEL_DIR+"test_sample",test_sample)
np.save(MODEL_DIR+"test_dataset",model_sample)
#calculate log likelihood of the dataset later on 


