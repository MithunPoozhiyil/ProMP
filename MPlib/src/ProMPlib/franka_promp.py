import numpy as np
import phase as phase
import basis as basis
import promps as promps
import tf.transformations as tf_tran
import matplotlib.pyplot as plt
import franka_kinematics
from mpl_toolkits.mplot3d import Axes3D

with open('/home/mithun/promp_codes/MPlib/src/100demos.npz', 'r') as f:
    data = np.load(f)
    Q = data['Q']
    time = data['time']

franka_kin = franka_kinematics.FrankaKinematics()

################################################
# To plot demonstrated end-eff trajectories

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for i in range(len(Q)):
#     endEffTraj = franka_kin.fwd_kin_trajectory(Q[i])
#     ax.scatter(endEffTraj[:,0], endEffTraj[:,1], endEffTraj[:,2], c='b', marker='.')
# plt.title('EndEff')

######################################
# To plot demonstrated trajectories Vs time

# for plotDoF in range(7):
#     plt.figure()
#     for i in range(len(Q)):
#         plt.plot(time[i] - time[i][0], Q[i][:, plotDoF])
#
#     plt.title('DoF {}'.format(plotDoF))
# plt.xlabel('time')
# plt.title('demonstrations')

############################################

phaseGenerator = phase.LinearPhaseGenerator()
basisGenerator = basis.NormalizedRBFBasisGenerator(phaseGenerator, numBasis=5, duration=1, basisBandWidthFactor=3,
                                                   numBasisOutside=1)
time_normalised = np.linspace(0, 1, 100)
nDof = 7
proMP = promps.ProMP(basisGenerator, phaseGenerator, nDof)
plotDof = 2

################################################################
# Conditioning in JointSpace

desiredTheta = np.array([0.5, 0.7, 0.5, 0.2, 0.6, 0.8, 0.1])
desiredVar = np.eye(len(desiredTheta)) * 0.0001
meanTraj, covTraj = proMP.getMeanAndCovarianceTrajectory(time_normalised)
newProMP = proMP.jointSpaceConditioning(0.5, desiredTheta=desiredTheta, desiredVar=desiredVar)
trajectories = newProMP.getTrajectorySamples(time_normalised, 4)
plt.figure()
plt.plot(time_normalised, trajectories[:, plotDof, :])
plt.xlabel('time')
plt.title('Joint-Space conditioning')
newProMP.plotProMP(time_normalised, [3, 4])

##################################################
# Conditioning in Task Space

learnedProMP = promps.ProMP(basisGenerator, phaseGenerator, nDof)
learner = promps.MAPWeightLearner(learnedProMP)
learner.learnFromData(Q, time)
mu_theta, sig_theta = learnedProMP.getMeanAndCovarianceTrajectory(np.array([1.0]))
sig_theta = np.squeeze(sig_theta)
mu_x = np.array([0.6, 0.5, 0.8])
sig_x = np.eye(3) * 0.0000002
q_home = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3]
T_desired, tmp = franka_kin.fwd_kin(q_home)
mu_ang_euler_des = tf_tran.euler_from_matrix(T_desired, 'szyz')
sig_euler = np.eye(3) * 0.0002
post_mean_Ash, post_cov = franka_kin.inv_kin_ash_pose(np.squeeze(mu_theta), sig_theta, mu_x, sig_x, mu_ang_euler_des, sig_euler)
taskProMP = learnedProMP.jointSpaceConditioning(1.0, desiredTheta=post_mean_Ash, desiredVar=post_cov)
trajectories_task_conditioned = taskProMP.getTrajectorySamples(time_normalised, 20)
plt.figure()
plt.plot(time_normalised, trajectories[:, plotDof, :])
plt.xlabel('time')
plt.title('Task-Space conditioning')

##############################################
# Plot of end-effector trajectories

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(trajectories_task_conditioned.shape[2]):
    endEffTraj = franka_kin.fwd_kin_trajectory(trajectories_task_conditioned[:, :, i])
    ax.scatter(endEffTraj[:, 0], endEffTraj[:, 1], endEffTraj[:, 2], c='b', marker='.')
plt.xlabel('X')
plt.xlabel('Y')
plt.title('EndEff trajectories')

##################################################
# To save the task conditioned trajectories for playing back on robot

with open('traject_task_conditioned1.npz', 'w') as f:
    np.save(f, trajectories_task_conditioned)

##############################################
plt.show()
print('Finished')

