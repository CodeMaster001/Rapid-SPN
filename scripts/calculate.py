import os;
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
file_name_spn = sys.argv[1]
file_name_fspn = sys.argv[2]
file_name_spn_stl=sys.argv[3]
file_name_fspn_stl=sys.argv[4]

#--------------loading spn and fspn for cifar data

table_spn=pd.read_csv(file_name_spn,delimiter=',',names=['a','b','c','d','e'])
table_spn = np.array(table_spn.values).astype(float)
table_fspn=pd.read_csv(file_name_fspn,delimiter=',',names=['a','b','c','d','e'])
table_fspn = np.array(table_fspn.values).astype(float)



#--------------loading spn and fspn for stl
table_spn_stl=pd.read_csv(file_name_spn_stl,delimiter=',',names=['a','b','c','d','e'])
table_spn_stl = np.array(table_spn_stl.values).astype(float)
table_fspn_stl=pd.read_csv(file_name_fspn_stl,delimiter=',',names=['a','b','c','d','e'])
table_fspn_stl = np.array(table_fspn_stl.values).astype(float)


#----------------completd loading both tables ----------------------------

col_index = [10,20,40,100,200,300,400,500,800,1000]
fig,axs = plt.subplots(1,4)
LEGEND_LOC = "upper left"
axs[0].plot(col_index,table_spn[:,0], label="SPN CIFAR LL")
axs[0].plot(col_index,table_fspn[:,1], label="FriendSPN CIFAR LL")
axs[1].plot(col_index,table_spn[:,2], label="SPN CIFAR TIME")
axs[1].plot(col_index,table_fspn[:,3], label="FriendSPN CIFAR TIME")
axs[2].plot(col_index,table_spn_stl[:,0], label="SPN STL LL")
axs[2].plot(col_index,table_fspn_stl[:,1], label="FriendSPN STL LL")
axs[3].plot(col_index,table_spn_stl[:,2], label="SPN STL TIME")
axs[3].plot(col_index,table_fspn_stl[:,3], label="FriendSPN STL TIME")
axs[0].legend(loc=LEGEND_LOC)
axs[1].legend(loc=LEGEND_LOC)
axs[2].legend(loc=LEGEND_LOC)
axs[3].legend(loc=LEGEND_LOC)
plt.show()
