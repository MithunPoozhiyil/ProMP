import numpy as np
import phase as phase
import basis as basis
import promps as promps
import tf.transformations as tf_tran
import matplotlib.pyplot as plt
from franka_kinematics import FrankaKinematics
from chomp import TrajectoryOptimize
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as opt
import scipy.ndimage
import time as tme
from scipy.spatial import distance
from numpy.lib.stride_tricks import as_strided

# with open('/home/ash/Ash/scripts/TrajectoryRecorder/Trajectories_bag/format/100demos.npz', 'r') as f:
# with open('/home/ash/Ash/scripts/TrajectoryRecorder/Trajectories_bag/tableandchar/format/vertical2bottle.npz', 'r') as f:
# with open('/home/automato/Ash/MPlib/src/trajecV2B.npz', 'r') as f:
# with open('/home/automato/Ash/MPlib/src/going_under_table_V2F.npz',
#               'r') as f:
with open('/home/automato/Ash/MPlib/src/going_under_table_together.npz',
          'r') as f:
    data = np.load(f)
    Q = data['Q']
    timeData = data['time']
A = []
for i in range(len(Q)):   # Q has 9 joints including 2 gripper joints, so making it 7
    A.append(Q[i][:, :-2])

class optim_weight_space(object):
    def __init__(self, time, nDoF, nBf, mu_x=[0.543602, -0.239265, 0.470648], sig_x=np.eye(3) * 0.0000002,
                 mu_quat=[0.724822, -0.282179, 0.544974, -0.313067], sig_quat=np.eye(4) * 0.0002,
                 obstacle_list=None, curr_jvs=None, QLists=None, timeLists=None):
        self.time_normalised = time
        self.nDoF = nDoF
        if curr_jvs is None:
            self.curr_jvs = Q[2][0, :-2]  # np.array([-0.25452656, -0.3225854, -0.24426211, -0.3497535 ,  0.06971882,  0.12819999,  0.65348821])
        else:
            self.curr_jvs = curr_jvs
        self.obstacle_list = obstacle_list
        self.franka_kin = FrankaKinematics()

        self.mu_x = mu_x
        self.sig_x = sig_x

        self.mu_quat_des = mu_quat
        self.sig_quat = sig_quat

        self.collision_threshold = 0.07
        # self.obstacle_list = np.array([[0.25, 0.0, 0.71, 0.1], [-0.15, 0.21, 0.51, 0.05]])  # for two spheres
        self.phaseGenerator = phase.LinearPhaseGenerator()
        #self.phaseGenerator = phase.SmoothPhaseGenerator(duration = 1)
        self.basisGenerator = basis.NormalizedRBFBasisGenerator(self.phaseGenerator, numBasis=nBf, duration=1,
                                                           basisBandWidthFactor=3,
                                                           numBasisOutside=1)
        self.basisMultiDoF = self.basisGenerator.basisMultiDoF(self.time_normalised, self.nDoF)
        self.learnedProMP = promps.ProMP(self.basisGenerator, self.phaseGenerator, self.nDoF)
        self.learner = promps.MAPWeightLearner(self.learnedProMP)
        self.Q_parsed, self.timeP = self.Q_parser(self.mu_x)
        # self.learnedData = self.learner.learnFromData(self.Q_parsed, self.timeP)
        if QLists is None:
            QLists = A
        else:
            QLists = QLists
        if timeLists is None:
            timeList = timeData
        else:
            timeList = timeLists
        self.learnedData = self.learner.learnFromData(QLists, timeList)

        self.mu_theta, self.sig_theta = self.learnedProMP.getMeanAndCovarianceTrajectory(np.array([1.0]))
        self.sig_theta = np.squeeze(self.sig_theta)

        self.post_mean_q, self.post_cov_q = self.franka_kin.inv_kin_ash_pose_quaternion(np.squeeze(self.mu_theta),
                                                self.sig_theta, self.mu_x, self.sig_x, self.mu_quat_des, self.sig_quat)
        self.taskProMP0 = self.learnedProMP.jointSpaceConditioning(0, desiredTheta=self.curr_jvs, desiredVar=np.eye(len(self.curr_jvs)) * 0.00001)
        self.taskProMP = self.taskProMP0.jointSpaceConditioning(1.0, desiredTheta=self.post_mean_q, desiredVar=np.eye(len(self.curr_jvs)) * 0.00001) #desiredVar=self.post_cov_q)

        self.trajectories_learned = self.learnedProMP.getTrajectorySamples(self.time_normalised, n_samples=20)
        self.trajectories_task_conditioned = self.taskProMP.getTrajectorySamples(self.time_normalised, n_samples=20)

        self.mean_cond_weight = self.taskProMP.mu
        self.covMat_cond_weight = self.taskProMP.covMat
        self.inv_covMat_cond = np.linalg.inv(self.covMat_cond_weight)

        self.trajectoryFlat = self.basisMultiDoF.dot(self.mean_cond_weight)
        self.trajectory = (self.trajectoryFlat.reshape((self.nDoF, self.trajectoryFlat.shape[0] / self.nDoF))).T
        # self.initial_joint_values = self.trajectory[0, :]
        self.initial_joint_values = self.curr_jvs  #Q[2][0, :-2]  #  np.array([-0.25452656, -0.3225854, -0.24426211, -0.3497535 ,  0.06971882,  0.12819999,  0.65348821])
        self.desired_joint_values = self.trajectory[-1, :]    # same as self.post_mean_q

        self.reguCoeff = 0.003
        self.lamda_smooth, self.lamda_obs = 10.0, 20.0
        self.optCurve, self.costs = [], []
        self.step_size = 0.06

    def Q_parser(self, mu_x):
        QQ, tme = [], []
        for i in range(len(Q)):
            jvs = Q[i][-1, :]
            T, _ = self.franka_kin.fwd_kin(jvs)
            endEffPos = T[:3, 3]
            dist = np.linalg.norm((mu_x - endEffPos))
            if dist <= 0.5:
                QQ.append(Q[i])
                tme.append(timeData[i])
        return QQ, tme

    def sphere(self, ax, radius, centre):
        u = np.linspace(0, 2 * np.pi, 13)
        v = np.linspace(0, np.pi, 7)
        x = centre[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = centre[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = centre[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
        xdata = scipy.ndimage.zoom(x, 3)
        ydata = scipy.ndimage.zoom(y, 3)
        zdata = scipy.ndimage.zoom(z, 3)
        ax.plot_surface(xdata, ydata, zdata, rstride=3, cstride=3, color='w', shade=0)

    def smoothness_objective(self, trajectoryFlat):   # used only to find the smoothness cost of traj sampled from ProMP
        trajectoryFlat = np.squeeze(trajectoryFlat)
        trajectory = trajectoryFlat.reshape((self.nDoF, -1)).T
        dim = np.multiply(*trajectory.shape)
        fd_matrix, b = self.finite_diff_matrix(trajectory)
        trajectory = trajectory.T.reshape((dim, 1))
        F_smooth = 0.5*(np.dot(trajectory.T, np.dot(fd_matrix, trajectory)) + np.dot(trajectory.T, b) + 0.25*np.dot(b.T, b))
        return F_smooth

    def promp_sampled_trajectory_cost(self, trajectories):  # trajectories is 3D array of (time x nDoF x nSamples)
        s = trajectories.shape
        smoothness_cost_chomp = np.zeros(s[2])   #  smoothness cost as described in chomp paper
        obstacle_cost = np.zeros(s[2])
        for i in range(s[2]):
            trajectory = trajectories[:, :, i]
            trajectoryFlat = trajectory.T.reshape((-1))
            smoothness_cost_chomp[i] = self.smoothness_objective(trajectoryFlat)
            if self.obstacle_list:
                obstacle_cost[i] = self.obstacle_cost(trajectoryFlat, self.obstacle_list)
        return smoothness_cost_chomp, obstacle_cost

    def optimise(self, weights, withGrad=True):

        if not withGrad:
        ################## Without Gradient #########################
            start = tme.time()
            optimized_weights = opt.minimize(self.calculate_total_cost, weights, method='BFGS',
                                            options={'maxiter': 150, 'disp': True},
                                            callback=self.optim_callback)
            total_time = tme.time() - start
            print 'ProMP time:without gradient:', total_time
            trajectoryFlat = self.basisMultiDoF.dot(optimized_weights.x)
            optimized_trajectory = trajectoryFlat.reshape((self.nDoF, -1)).T
            optimized_trajectory = optimized_trajectory[1:-1]
            optimized_trajectory = np.insert(optimized_trajectory, 0, self.initial_joint_values, axis=0)
            optimized_trajectory = np.insert(optimized_trajectory, optimized_trajectory.shape[0],
                                            self.desired_joint_values, axis=0)

        else:
            # ################### With Gradient ############################
            start = tme.time()
            optimized_weights = opt.minimize(self.calculate_total_cost, weights, method='BFGS', jac=self.cost_gradient_analytic,
                                                  options={'maxiter': 150, 'disp': True, 'gtol': 1e-6}, callback=self.optim_callback)
            total_time = tme.time() - start
            print 'ProMP time:with gradient:', total_time

            trajectoryFlat = self.basisMultiDoF.dot(optimized_weights.x)
            optimized_trajectory = (trajectoryFlat.reshape((self.nDoF, -1))).T
            optimized_trajectory = optimized_trajectory[1:-1]
            optimized_trajectory = np.insert(optimized_trajectory, 0, self.initial_joint_values, axis=0)
            optimized_trajectory = np.insert(optimized_trajectory, optimized_trajectory.shape[0], self.desired_joint_values, axis=0)

        return optimized_trajectory, optimized_weights.x, self.costs, optimized_weights.success

    def calculate_jacobian(self, robot_body_points, joint_index, joint_values=None):
        class ParentMap(object):
            def __init__(self, num_joints):
                self.joint_idxs = [i for i in range(num_joints)]
                self.p_map = np.zeros((num_joints, num_joints))
                for i in range(num_joints):
                    for j in range(num_joints):
                        if j <= i:
                            self.p_map[i][j] = True

            def is_parent(self, parent, child):
                if child not in self.joint_idxs or parent not in self.joint_idxs:
                    return False
                return self.p_map[child][parent]

        parent_map = ParentMap(7)

        if joint_values is None:
            joint_values = list(self.robot_state.get_current_state().joint_state.position)[:7]

        _, t_joints = self.franka_kin.fwd_kin(joint_values)
        joint_axis = t_joints[:, :3, 2]
        joint_positions = t_joints[:7, :3, 3]  # Excluding the fixed joint at the end
        jacobian = list()
        for i, points in enumerate(np.split(robot_body_points, joint_index)[:-1]):
            # print i, len(points)
            if i == 0:
                for point in points:
                    jacobian.append(np.zeros((3, 7)))
            else:
                for point in points:
                    jacobian.append(np.zeros((3, 7)))
                    for joint in parent_map.joint_idxs:
                        if parent_map.is_parent(joint, i-1):
                            # find cross product
                            col = np.cross(joint_axis[joint, :], point-joint_positions[joint])
                            jacobian[-1][0][joint] = col[0]
                            jacobian[-1][1][joint] = col[1]
                            jacobian[-1][2][joint] = col[2]
                        else:
                            jacobian[-1][0][joint] = 0.0
                            jacobian[-1][1][joint] = 0.0
                            jacobian[-1][2][joint] = 0.0
        return np.array(jacobian)

    def finite_diff_matrix(self, trajectory):
        rows, columns = trajectory.shape  # columns = nDoF
        A = 2 * np.eye(rows)
        A[0, 1] = -1
        A[rows-1, rows-2] = -1
        for ik in range(0, rows-2):
            A[ik + 1, ik] = -1
            A[ik + 1, ik + 2] = -1

        dim = rows*columns
        fd_matrix = np.zeros((dim, dim))
        b = np.zeros((dim, 1))
        i, j = 0, 0
        while i < dim:
            fd_matrix[i:i+len(A), i:i+len(A)] = A
            b[i] = -2 * self.initial_joint_values[j]
            b[i+len(A)-1] = -2 * self.desired_joint_values[j]
            i = i + len(A)
            j = j + 1
        return fd_matrix, b

    def get_robot_discretised_points(self, fwd_k_j_positions, step_size=0.2):
        discretised = list()
        j_index = list()
        # self.get_robot_discretised_joint_index(fwd_k_j_positions, step_size=0.2)
        for j in range(len(fwd_k_j_positions) - 1):
            w = fwd_k_j_positions[j+1] - fwd_k_j_positions[j]
            if len(w[w != 0.]):
                step = step_size * w / np.linalg.norm(w)
                n = int(np.linalg.norm(w) / np.linalg.norm(step)) + 1
                discretised.extend(np.outer(np.arange(1, n), step) + fwd_k_j_positions[j])
            j_index.append(len(discretised))
        self.j_index = j_index
        return np.array(discretised)

    def get_robot_joint_positions(self, joint_values):
        _, t_joints = self.franka_kin.fwd_kin(joint_values)
        return np.vstack(([0., 0., 0.], t_joints[:, :3, 3]))

    def calculate_robot_body_points(self, trajectory):
        # trajectory = np.vstack((trajectory, self.desired_joint_values))
        return np.array([self.get_robot_discretised_points(self.get_robot_joint_positions(joint_values), self.step_size,
                                                           ) for joint_values in trajectory])

    def calculate_normalised_workspace_velocity(self, robot_body_points):
        velocity = np.gradient(robot_body_points, axis=0)
        vel_magnitude = np.linalg.norm(velocity, axis=2)
        vel_normalised = np.divide(velocity, vel_magnitude[:, :, None], out=np.zeros_like(velocity), where=vel_magnitude[:, :, None] != 0)
        return vel_normalised, vel_magnitude, velocity

    def smoothness_cost(self, weights):
        temp1 = (weights - self.mean_cond_weight).reshape(-1, 1)
        temp2 = temp1.T.dot(self.inv_covMat_cond)
        return 0.5 * temp2.dot(temp1) + 0.5 * self.reguCoeff * weights.dot(weights)

    def obstacle_cost(self, trajectoryFlat, obstacle_list,):
        trajectory = trajectoryFlat.reshape((self.nDoF, -1)).T
        # trajectory = trajectory[1:-1]
        robot_body_points = self.calculate_robot_body_points(trajectory)
        vel_normalised, vel_mag, vel = self.calculate_normalised_workspace_velocity(robot_body_points)
        dist = self.compute_minimum_distance_to_objects(robot_body_points, obstacle_list)
        obstacle_cost_potential = np.array(self.cost_potential(dist))
        return np.sum(np.multiply(obstacle_cost_potential, vel_mag))

    def calculate_total_cost(self, weights):
        weights = np.squeeze(weights)
        trajectoryFlat = self.basisMultiDoF.dot(weights)
        smoothness_cost = self.smoothness_cost(weights)  # smoothness of trajectory is captured here
        if self.obstacle_list is not None:
            obstacle_cost = self.obstacle_cost(trajectoryFlat, self.obstacle_list)
        else:
            obstacle_cost = 0.0
        # return 0.001 * smoothness_cost + 1.5 * obstacle_cost   #  for without gradient
        return self.lamda_smooth * smoothness_cost + self.lamda_obs * obstacle_cost    #  for with gradient

    def smoothness_gradient(self, weights):
        return self.inv_covMat_cond.dot(weights - self.mean_cond_weight) + self.reguCoeff * weights

    def compute_minimum_distance_to_objects(self, robot_body_points, obstacle_list, body_sphere_size=0.12):
        rbp_shape = robot_body_points.shape
        return np.min(
            distance.cdist(
                robot_body_points.reshape(-1, 3, order='A'),
                obstacle_list[:, :3]
            ) - obstacle_list[:, 3] - body_sphere_size,
            axis=1
        ).reshape(rbp_shape[:2])

    def cost_potential(self, D, collision_threshold=0.07):
        c = np.zeros(D.shape)
        return np.where(
            D < 0,
            -D + 0.5 * collision_threshold,
            np.where(
                D <= collision_threshold,
                (0.5 * (D - collision_threshold) ** 2) / collision_threshold,
                c
            )
        )

    def fun(self, points):
        dist = self.compute_minimum_distance_to_objects(points.reshape((1, 1, 3)), self.obstacle_list)
        return np.array(self.cost_potential(dist))

    def gradient_cost_potential(self, robot_discretised_points):
        gradient_cost_potential = np.zeros(robot_discretised_points.reshape(-1, 3).shape)
        for i, points in enumerate(robot_discretised_points.reshape((-1, 3))):
            gradient_cost_potential[i, :] = opt.approx_fprime(points, self.fun, [1e-06, 1e-06, 1e-06])
        return np.array(gradient_cost_potential).reshape(robot_discretised_points.shape[0], -1, 3)

    def extract_block_diag(self, A, M=3, k=0):
        ny, nx = A.shape
        ndiags = min(map(lambda x: x // M, A.shape))
        offsets = (nx * M + M, nx, 1)
        strides = map(lambda x: x * A.itemsize, offsets)
        if k > 0:
            B = A[:, k * M]
            ndiags = min(nx // M - k, ny // M)
        else:
            k = -k
            B = A[k * M]
            ndiags = min(nx // M, ny // M - k)
        return as_strided(B, shape=(ndiags, M, M),
                          strides=((nx * M + M) * A.itemsize, nx * A.itemsize, A.itemsize))

    def calculate_curvature(self, vel_normalised, vel_magnitude, velocity):
        s = vel_normalised.shape
        acceleration = np.gradient(velocity, axis=0).reshape(-1, 3)
        ttm = vel_normalised.reshape(-1, 1).dot(vel_normalised.reshape(1, -1))
        orthogonal_projector = self.extract_block_diag((np.eye(ttm.shape[0]) - ttm), 3)
        temp = np.array([orthogonal_projector[i, :, :].dot(acceleration[i, :]) for i in range(orthogonal_projector.shape[0])])
        curvature = np.divide(temp, vel_magnitude.reshape(-1, 1)**2, out=np.zeros_like(temp), where=vel_magnitude.reshape(-1, 1) !=0)
        return curvature.reshape(s), orthogonal_projector.reshape((s[0], s[1], s[2], s[2]))

    def calculate_obstacle_cost_gradient(self, trajectoryFlat):
        trajectory = np.squeeze(trajectoryFlat).reshape((self.nDoF, -1)).T
        # trajectory = trajectory[1:-1]
        robot_discretised_points = self.calculate_robot_body_points(trajectory)
        vel_normalised, vel_magnitude, velocity = self.calculate_normalised_workspace_velocity(robot_discretised_points)
        curvature, orthogonal_projector = self.calculate_curvature(vel_normalised, vel_magnitude, velocity)
        dist = self.compute_minimum_distance_to_objects(robot_discretised_points, self.obstacle_list,)
        obstacle_cost_potential = np.array(self.cost_potential(dist))
        gradient_cost_potential = self.gradient_cost_potential(robot_discretised_points)
        obstacle_gradient = np.zeros((trajectory.shape[0], self.nDoF))
        a = robot_discretised_points  # [:-1]
        for jvs in range(trajectory.shape[0]):
            obst_grad = np.zeros(trajectory.shape[1])
            jacobian = self.calculate_jacobian(a[jvs], self.j_index, trajectory[jvs])
            for num_points in range(robot_discretised_points.shape[1]):
                temp1 = orthogonal_projector[jvs, num_points].dot(gradient_cost_potential[jvs, num_points, :])
                temp2 = obstacle_cost_potential[jvs, num_points] * curvature[jvs, num_points, :]
                temp3 = vel_magnitude[jvs, num_points] * (temp1 - temp2)
                obst_grad += jacobian[num_points, :].T.dot(temp3)
                # obst_grad1 += self.basisMultiDoFParsed[jvs].T.dot(jacobian[num_points, :].T.dot(temp3))
            obstacle_gradient[jvs, :] = obst_grad
        return obstacle_gradient.T

    def cost_gradient_analytic(self, weights):  # calculate grad(cost) = grad(smoothness_cost) + grad(obstacle_cost)
        weights = np.squeeze(weights)
        trajectoryFlat = self.basisMultiDoF.dot(weights)
        smoothness_gradient = self.smoothness_gradient(weights)
        if self.obstacle_list is not None:
            obstacle_gradient_traj = self.calculate_obstacle_cost_gradient(trajectoryFlat)
            obstacle_gradient_weight = self.basisMultiDoF.T.dot(obstacle_gradient_traj.reshape(-1))
        else:
            obstacle_gradient_weight = np.zeros(len(weights))
        # trajectory = np.squeeze(trajectory)
        cost_gradient = self.lamda_obs * obstacle_gradient_weight + self.lamda_smooth * smoothness_gradient
        return np.squeeze(cost_gradient)

    def optim_callback(self, xk):
        self.costs.append(self.calculate_total_cost(xk)[0, 0])
        self.optCurve.append(xk)
        print 'Iteration {}: {:2.4}\n'.format(len(self.optCurve), self.costs[len(self.optCurve) - 1])

    def animation(self, optimized_trajectory, initial_trajectory):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        # ax.axis('off')
        plt.show(block=False)
        while True:
            for i in range(len(optimized_trajectory)):
                _, T_joint_optim = self.franka_kin.fwd_kin(optimized_trajectory[i])
                _, T_joint_init = self.franka_kin.fwd_kin(initial_trajectory[i])
                ax.clear()
                if self.obstacle_list is not None:
                    for object in self.obstacle_list:
                        self.sphere(ax, object[-1], object[0:-1])
                self.franka_kin.plotter(ax, T_joint_optim, 'optim', color='blue')
                self.franka_kin.plotter(ax, T_joint_init, 'init', color='red')
                # for x, y, z in self.get_robot_discretised_points(trajectory[i],step_size=0.2):
                #     plt.grid()
                #     ax.scatter(x, y, z, 'gray')
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.01)
            plt.pause(1)
        # plt.show(block=True)

    def discretize(self, start, goal, step_size=0.02):
        w = goal - start
        step = step_size * w / np.linalg.norm(w)
        n = int(np.linalg.norm(w) / np.linalg.norm(step)) + 1
        discretized = np.outer(np.arange(1, n), step) + start
        return discretized


if __name__ == '__main__':
    time = np.linspace(0, 1, 15)
    nDof, nBf = 7, 5
    tr_opt = TrajectoryOptimize(nDof)
    obstacle_list = np.array([[0.15, -0.071, 0.79, 0.05], [-0.15, 0.21, 0.51, 0.05]])
    obstacle_list_fromRviz = tr_opt.populate_obstacles()
    print obstacle_list_fromRviz.shape
    tme.sleep(1)
    # weight_space_optim = optim_weight_space(time, nDof, nBf)
    weight_space_optim = optim_weight_space(time, nDof, nBf, obstacle_list=obstacle_list)#_fromRviz)
    mu_w = weight_space_optim.mean_cond_weight
    cov_w = weight_space_optim.covMat_cond_weight
    weights = np.random.multivariate_normal(mu_w, cov_w)
    # weights = np.random.multivariate_normal(mu_w, np.eye(len(mu_w)) * 0.00001)  # [nBf:-nBf]

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # traj_task = weight_space_optim.trajectories_task_conditioned
    # trajectories_learned = weight_space_optim.trajectories_learned

    # for i in range(traj_task.shape[2]):
    #     endEffTraj = weight_space_optim.franka_kin.fwd_kin_trajectory(traj_task[:, :, i])
    #     ax.scatter(endEffTraj[:, 0], endEffTraj[:, 1], endEffTraj[:, 2], c='b')
    #     endEffTraj = weight_space_optim.franka_kin.fwd_kin_trajectory(trajectories_learned[:, :, i])
    #     ax.scatter(endEffTraj[:, 0], endEffTraj[:, 1], endEffTraj[:, 2], c='r')
    #
    # for i in range(20):
    #     endEffTraj = weight_space_optim.franka_kin.fwd_kin_trajectory(Q[i])
    #     ax.scatter(endEffTraj[:, 0], endEffTraj[:, 1], endEffTraj[:, 2], c='g')
    # plt.xlabel('X')
    # plt.xlabel('Y')
    # plt.title('EndEff trajectories')

    # plt_dof = [0, 1, 2, 3, 4, 5, 6,]
    # for j in plt_dof:
    #     plt.figure()
    #     for i in range(traj_task.shape[2]):
    #         plt.plot(time, traj_task[:, j, i])


    # smoothness_cost = weight_space_optim.smoothness_cost(weights)
    # obstacle_cost = weight_space_optim.obstacle_cost(weights)
    initial_joint_values = weight_space_optim.initial_joint_values
    desired_joint_values = weight_space_optim.desired_joint_values
    n = np.linalg.norm((desired_joint_values - initial_joint_values))/(len(time)-1)

    initial_trajectory = weight_space_optim.discretize(initial_joint_values, desired_joint_values, step_size=n)

    initial_trajectory = np.insert(initial_trajectory, len(initial_trajectory), desired_joint_values,
                                   axis=0)  # inserting goal state at the end of discretized
    initial_trajectory = np.insert(initial_trajectory, 0, initial_joint_values, axis=0)

    initial_trajectory = weight_space_optim.trajectory
    optimised_trajectory, _, _, success = weight_space_optim.optimise(weights, withGrad=True)
    weight_space_optim.animation(optimised_trajectory, initial_trajectory)


    print 'hi'


