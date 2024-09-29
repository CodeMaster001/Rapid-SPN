
import numpy as np
import numpy
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn import preprocessing
from spatialtree import *;
from libsvm.svmutil import *
from spn.structure.Base import *
import numpy as np, numpy.random
import logging
import numpy as np
import subprocess
#tf.logging.set_verbosity(tf.logging.INFO)
logging.getLogger().setLevel(logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
np.random.seed(42)
def run_execution_float(X_train,X_test, min_instance_slice, height, leaves_size, threshold, output_file_name):
    train_set = []
    test_set = []
    opt_args= str(output_file_name) + ' ' + str(min_instance_slice) + ' ' + str(height) +' '+ str(leaves_size) + ' ' +str(threshold)
    context=[]
    for i in range(0,X_train.shape[1]):
        context.append(Gaussian)
#X_train,X_test=train_dataset_df.values[train_index],train_dataset_df.values[test_index]
    X=numpy.nan_to_num(X_train)
    X = preprocessing.normalize(X, norm='l2')
    X_test = numpy.nan_to_num(X_test)
    X_test = preprocessing.normalize(X_test, norm='l2')
    X_test =X_test.astype(numpy.float32)
    train_set.append(X)
    test_set.append(X_test)
    np.save('train', X)
    np.save("test",X_test)
    np.save("context",context)
    P=subprocess.Popen(['python3 experiment.py train.npy test.npy context.npy '+opt_args.strip()],shell=True)
    P.communicate()
    P.wait();
    P.terminate()
    print("process completed")
    
def run_execution_character(X_train,X_test, min_instance_slice, height, leaves_size, threshold, output_file_name):
    train_set = []
    test_set = []
    opt_args= str(output_file_name) + ' ' + str(min_instance_slice) + ' ' + str(height) +' '+ str(leaves_size) + ' ' +str(threshold)
    context=[]
    for i in range(0,X_train.shape[1]):
        context.append(Categorical)
    train_set.append(X_train)
    test_set.append(X_test)
    np.save('train', X_train)
    np.save("test",X_test)
    np.save("context",context)
    P=subprocess.Popen(['python3 experiment.py train.npy test.npy context.npy '+opt_args.strip()],shell=True)
    P.communicate()
    P.wait();
    P.terminate()
    print("process completed")