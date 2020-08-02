from matplotlib import pyplot as plt;
import numpy as np
import pandas as pd;
import seaborn
import seaborn as sns; sns.set()
data = pd.read_csv("data.csv",delimiter=",").values;
print(data)
plt.scatter(data[:,3],data[:,5]/1000,label='ue')
plt.scatter(data[:,3],data[:,7]/1000,label="SP")
plt.xlabel('Feature Size')
plt.ylabel('Time(x1000) sec')
plt.gca().legend(('SPN Time','SPNRP Time'))
plt.show()