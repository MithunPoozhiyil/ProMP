import numpy as np
from franka_kinematics import FrankaKinematics
import tf.transformations as tf_tran


with open('/home/ash/Ash/scripts/TrajectoryRecorder/Trajectories_bag/tableandchar/format/bottle2cup.npz', 'r') as f:
    data = np.load(f)
    Q = data['Q']
    timeData = data['time']

nDoF = 9
franka_kin = FrankaKinematics()
# end_robot_config = np.zeros((len(Q), nDoF))
euler, position, quaternion = [], [], []
for i in range(len(Q)):
    end_robot_config = Q[i][-1]
    T, _ = franka_kin.fwd_kin(end_robot_config[:-2])
    euler.append(tf_tran.euler_from_matrix(T))
    position.append(T[:3, -1])
    quaternion.append(tf_tran.quaternion_from_matrix(T))

euler = np.array(euler)
position = np.array(position)
quaternion = np.array(quaternion)

# euler_mean, euler_std = np.mean(euler, axis=0), np.std(euler, axis=0)
# position_mean, position_std = np.mean(position, axis=0), np.std(position, axis=0)
# quaternion_mean, quaternion_std = np.mean(quaternion, axis=0), np.std(quaternion, axis=0)
#
# pos_cov = np.cov(position, rowvar=0)
# euler_cov = np.cov(euler, rowvar=0)
# nSample = 100
# posSampled = np.random.multivariate_normal(position_mean, pos_cov, nSample)
# EulerSampled = np.random.multivariate_normal(euler_mean, euler_cov, nSample)
# QuartSampled = np.zeros((EulerSampled.shape[0], 4))
# for i in range(EulerSampled.shape[0]):
#     QuartSampled[i, :] = tf_tran.quaternion_from_euler(EulerSampled[i][0], EulerSampled[i][1], EulerSampled[i][2])

# euler_sampleX, euler_sampleY, euler_sampleZ = np.random.normal(euler_mean[0], euler_std[0], nSample), \
#                                               np.random.normal(euler_mean[1], euler_std[1], nSample), \
#                                               np.random.normal(euler_mean[2], euler_std[2], nSample)
#
# posi_sampleX, posi_sampleY, posi_sampleZ = np.random.normal(position_mean[0], position_std[0], nSample), \
#                                               np.random.normal(position_mean[1], position_std[1], nSample), \
#                                               np.random.normal(position_mean[2], position_std[2], nSample)
#
# Quart_sampleX, Quart_sampleY, Quart_sampleZ, Quart_sampleW = np.random.normal(quaternion_mean[0], quaternion_std[0], nSample), \
#                                               np.random.normal(quaternion_mean[1], quaternion_std[1], nSample), \
#                                               np.random.normal(quaternion_mean[2], quaternion_std[2], nSample), \
#                                               np.random.normal(quaternion_mean[2], quaternion_std[2], nSample)

# position_sample = np.vstack((posi_sampleX, posi_sampleY, posi_sampleZ)).T
# euler_sample = np.vstack((euler_sampleX, euler_sampleY, euler_sampleZ)).T
# quart_sample = np.vstack((Quart_sampleX, Quart_sampleY, Quart_sampleZ, Quart_sampleW)).T

np.savetxt('positionSampleB2Cdel.txt', position, fmt='%4f')
# np.savetxt('EulerSampleB2C.txt', euler, fmt='%4f')
np.savetxt('QuartSampleB2Cdel.txt', quaternion, fmt='%4f')

print 'hi'
