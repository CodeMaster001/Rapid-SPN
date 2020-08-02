import os;
import sys
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
sns.set()
iris_tab = pd.read_csv("results/iris.10.log",delimiter=",",names=["SPN-LL", "FriendSPN-LL","SPN-TIME", "FriendSPN-Time","Instance-slice"])
irish_tab = pd.read_csv("results/irish.10.log",delimiter=",",names=["SPN-LL", "FriendSPN-LL","SPN-TIME", "FriendSPN-Time","Instance-slice"])
balance_tab = pd.read_csv("results/balance_scale.10.log",delimiter=",",names=["SPN-LL", "FriendSPN-LL","SPN-TIME", "FriendSPN-Time","Instance-slice"])
hayes_tab = pd.read_csv("results/hayes.10.log",delimiter=",",names=["SPN-LL", "FriendSPN-LL","SPN-TIME", "FriendSPN-Time","Instance-slice"])
iris_tab["range"]=range(0,iris_tab.values.shape[0])
print(iris_tab)

idx = list(range(0,iris_tab.values.shape[0]))
print(idx)
f, axes = plt.subplots(2, 2)
sns.regplot(x=idx, y=iris_tab['SPN-LL'],ax=axes[0,0])
sns.regplot(x=idx, y=iris_tab['FriendSPN-LL'],ax=axes[0,0])
sns.regplot(x=idx, y=irish_tab['SPN-LL'],ax=axes[0,1])
sns.regplot(x=idx, y=irish_tab['FriendSPN-LL'],ax=axes[0,1])
splot=sns.regplot(x=idx, y=balance_tab['SPN-LL'],ax=axes[1,0])
sns.regplot(x=idx, y=balance_tab['FriendSPN-LL'],ax=axes[1,0])
splot=sns.regplot(x=idx, y=hayes_tab['SPN-LL'],ax=axes[1,1])
sns.regplot(x=idx, y=hayes_tab['FriendSPN-LL'],ax=axes[1,1])
axes[0,0].set_xlabel('')
axes[0,0].set_ylabel('')
axes[0,1].set_xlabel('')
axes[0,1].set_ylabel('')
axes[1,0].set_xlabel('')
axes[1,0].set_ylabel('')
axes[1,1].set_xlabel('')
axes[1,1].set_ylabel('')
plt.show()
