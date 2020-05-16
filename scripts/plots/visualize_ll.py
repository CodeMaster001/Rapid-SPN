from matplotlib import pyplot as plt;
import numpy as np
import pandas as pd;
import seaborn
import seaborn as sns; sns.set()
data = pd.read_csv("data.csv",delimiter=",").values;
print(data)
plt.scatter(data[:,3],data[:,4],label='ue')
plt.scatter(data[:,3],data[:,6],label="SP")
plt.xlabel('Feature Size')
plt.ylabel('Time(x10000) sec')
plt.gca().legend(('SPN LL','SPNRP LL'))
plt.show()