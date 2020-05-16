import os;
import sys
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
file_name = sys.argv[1]
f = open(file_name,'r')
table = list()
counter = 0;
for i in f:
	if counter%2!=0:
		table.append(i.strip().split(','))
	counter = counter + 1;
table = np.array(table).astype(float)

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
