
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
#tf.logging.set_verbosity(tf.logging.INFO)
logging.getLogger().setLevel(logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
from utils import run_execution_float


# experiment.py train.csv test.csv context.npy instance_slice epochs height prob leaves_size
train_dataset,labels= fetch_openml(name='hayes-roth', version=1,return_X_y=True)
train_dataset_df = pd.DataFrame(train_dataset)
print(train_dataset_df.head())
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
output_file_name='hayes.10.log'
min_instances_slice=10
epochs=8000
height=6
prob=0.4
leaves_size=4
threshold =0.4

opt_args=str(output_file_name) + ' ' + str(min_instances_slice) +' ' +str(height) + ' '+str(leaves_size)+' '+str(threshold)

for X_train,X_test in kf.split(train_dataset):
    run_execution_float(X_train=train_dataset.values[X_train],X_test=train_dataset.values[X_test],min_instance_slice=min_instance_slice,height=height,leaves_size=leaves_size,threshold=threshold,output_file_name=output_file_name)
       

print("process completed")
