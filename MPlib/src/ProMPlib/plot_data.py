import numpy as np

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

with open('/home/mithun/promp_codes/MPlib/src/100demos.npz', 'rb') as f:
    data = np.load(f)
    Q = data['Q']
    time = data['time']
################################################
#to plot demonstrated end-eff trajectories

# fig = plt.figure()

######################################
# To plot demonstrated trajectories Vs time

# for plotDoF in range(7):
plt.figure()
for i in range(len(Q)):
    plt.plot(time[i] - time[i][0], Q[i][:, 1])

plt.title('DoF {}'.format(2))
plt.xlabel('time')
plt.title('demonstrations')

############################################


plt.show()