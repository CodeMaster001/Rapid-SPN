import numpy as np
import matplotlib.pyplot as plt
import sys
ll_theirs = np.loadtxt(sys.argv[1],delimiter=',')[1]
time_theirs= np.loadtxt(sys.argv[2],delimiter=',')[1]
ours_ll= np.loadtxt(sys.argv[3],delimiter=',')
ours_time = np.loadtxt(sys.argv[4],delimiter=',')

fig, axes = plt.subplots(nrows=2, ncols=2)
axes[0,0].plot(range(0,len(ll_theirs)),ll_theirs, color='red',label="LEARNSPN LL")
axes[0,0].set_xlabel('Slice No')
axes[0,0].set_xlabel('LL')
axes[0,0].legend()
axes[0,1].plot(range(0,len(ours_ll)),ours_ll, color='blue',label='SPNRP LL')
axes[0,1].legend()
axes[1,0].plot(range(0,len(time_theirs)),time_theirs, color='red',label="LEARNSPN Time")
axes[1,0].legend()
axes[1,1].plot(range(0,len(ours_time)),ours_time, color='blue',label='SPNRP Time')
axes[1,1].legend()
plt.show()

print("--ll--")
print(ll_theirs)
print(ours_ll)
print(np.mean(ll_theirs))
print(np.var(ll_theirs))
print(np.mean(ours_ll))
print(np.var(ours_ll))
print("------")



print("--tt--")
print(np.mean(time_theirs))
print(np.var(time_theirs))
print(np.mean(ours_time))
print(np.var(ours_time))
print("------")
#plt.plot(time[0], ll[0],'o')
#plt.plot(time[1], ll[1],'o')
#plt.show()