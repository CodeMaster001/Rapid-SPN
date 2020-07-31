import os;
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
file_name = sys.argv[1]
table=pd.read_csv(file_name,delimiter=',',names=['a','b','c','d','e'])
table = np.array(table.values).astype(float)

mean_result = np.mean(table,axis=0)

std_result = np.std(table,axis=0)

print(str(mean_result[0])+':'+str(mean_result[1])+":"+str(mean_result[2])+":"+str(mean_result[3]))

print(str(std_result[0])+':'+str(std_result[1])+":"+str(std_result[2])+":"+str(std_result[3]))

spn_mean = table[:,0]
print(spn_mean.shape)
fig,ax = plt.subplots(1)
ax.plot(list(range(0,table.shape[0])),table[:,0], label="SPN LL")
ax.plot(list(range(0,table.shape[0])),table[:,1], label="FriendSPN LL")
plt.title(sys.argv[2])
ax.legend()
plt.show()
