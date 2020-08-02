import sys
from matplotlib import pyplot as plt
import pickle
from spn.algorithms.Sampling import sample_instances
import spn
import math
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from numpy.random.mtrand import RandomState
from sklearn.datasets import fetch_openml
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
train_dataset,labels= fetch_openml(name=sys.argv[1], version=1,return_X_y=True)
train_dataset_df = pd.DataFrame(train_dataset)
kf = KFold(n_splits=10,shuffle=True)

np.random.seed(42)
mse_spn=0;
mse_spn_friend=0;
mse_spn_avg=list();
mse_spn_friend_avg=list()
for train_index,test_index in kf.split(train_dataset_df):

	for i in range(1,10):
		X_train,X_test=train_dataset_df.values[train_index],train_dataset_df.values[test_index]       
		model_spn=open(sys.argv[2],'rb')
		model_friend=open(sys.argv[3],'rb')
		X_test = preprocessing.normalize(X_test, norm='l2') 
		model_spn = pickle.load(model_spn)
		model_friend=pickle.load(model_friend) 
		sample_holder=np.array([np.nan]*X_test.shape[0]*X_test.shape[1])
		sample_holder=sample_holder.reshape(X_test.shape[0],X_test.shape[1])
		spn_samples=sample_instances(model_spn,sample_holder, RandomState(42))
		spn_friend=sample_instances(model_friend,sample_holder, RandomState(42))
		mse_spn_current=mean_squared_error(X_test, spn_samples)
		mse_spn_friend_current=mean_squared_error(X_test, spn_friend)
		if i==1:
			mse_spn=mse_spn_current
			mse_spn_friend=mse_spn_friend_current
		mse_spn=min(mse_spn,mse_spn_current)
		mse_spn_friend=min(mse_spn_friend,mse_spn_friend_current)

	print(str(mse_spn)+':'+str(mse_spn_friend))
	mse_spn_avg.append(mse_spn)
	mse_spn_friend_avg.append(mse_spn_friend)
fig,ax = plt.subplots(1)
sns.set_style("darkgrid")
ax.plot(list(range(0,len(mse_spn_avg))),mse_spn_avg)
ax.plot(list(range(0,len(mse_spn_friend_avg))),mse_spn_friend_avg)
print(np.mean(mse_spn_avg))
print(np.mean(mse_spn_friend_avg))
plt.title(sys.argv[1])
plt.show();