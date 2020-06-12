#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import time
import scipy.ndimage
import scipy.optimize as opt
from scipy.spatial import distance
from franka_kinematics import FrankaKinematics
from trajectory_optimize import trajectory_optimization
import matplotlib.pyplot as plt
import moveit_commander
from mpl_toolkits.mplot3d import Axes3D
from numpy.lib.stride_tricks import as_strided


class TrajectoryOptimize(object):

    def __init__(self, nDoF, obstacle_list=None, ini_jv=None, des_jv=None,):
        super(TrajectoryOptimize, self).__init__()
        self.initial_joint_values = ini_jv
        self.desired_joint_values = des_jv
        self.scene = moveit_commander.PlanningSceneInterface()
        # if obstacle_list is None:
        #     self.obstacle_list = np.array([[0.25, 0.21, 0.71, 0.1], [-0.15, 0.21, 0.51, 0.05]])  # for two spheres
        # else:
        #     self.obstacle_list = obstacle_list
        self.obstacle_list = obstacle_list
        self.nDoF = nDoF
        self.franka_kin = FrankaKinematics()
        self.workspace_dim = (-0.5, -0.5, 0.2, 1.5, 0.5, 1.9)
        # self.obstacle_list = np.array([[0.25, 0.21, 0.71, 0.1]])  # for one sphere (x,y,z, radius)

        # self.obstacle_list = np.array([[0.25, 0.21, 0.71, 0.1],
        #                              [-0.15, 0.21, 0.51, 0.05],
        #                              [0.15, 0.61, 0.51, 0.05],
        #                              [-0.15, -0.61, 0.51, 0.05],
        #                              [-0.65, 0.31, 0.21, 0.05],
        #                              [-0.55, 0.71, 0.71, 0.06],
        #                              [-0.95, -0.51, 0.41, 0.07],
        #                              [-0.85, 0.27, 0.61, 0.08],
        #                              [-0.25, -0.31, 0.11, 0.04],
        #                              [-0.75, 0.71, 0.91, 0.09],
        #                              [-0.35, 0.51, 0.81, 0.02],
        #                              [-0.95, -0.81, 0.61, 0.03],
        #                              [-0.75, 0.11, 0.41, 0.02],
        #                              [-0.55, -0.61, 0.21, 0.06],
        #                              [0.25, 0.31, 0.71, 0.07],
        #                              [-0.35, -0.41, 0.41, 0.02],
        #                              [-0.25, -0.51, 0.61, 0.08],
        #                              ])  # for 15 spheres
        # self.obstacle_list1 = self.populate_obstacles()
        self.optCurve, self.costs = [], []
        self.step_size = 0.06
        self.debug = False

        self.lamda_smooth, self.lamda_obs = 10.0, 20.0

    def populate_obstacles(self,):
        # if the obstacle is within robot workspace, then consider those as obstacles
        obs = []
        x_min, x_max = self.workspace_dim[0], self.workspace_dim[3]
        y_min, y_max = self.workspace_dim[1], self.workspace_dim[4]
        z_min, z_max = self.workspace_dim[2], self.workspace_dim[5]

        for k, v in self.scene.get_objects().items():
            px, py, pz = v.primitive_poses[0].position.x, v.primitive_poses[0].position.y, v.primitive_poses[0].position.z
            if px >= x_min and px <=x_max:
                if py >= y_min and py <= y_max:
                    if pz >= z_min and pz <= z_max:
                        obs.append([px, py, pz, 0.05])
        return np.array(obs)

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

    def optimise(self, trajectory, withGrad=True):
        self.initial_joint_values = trajectory[0]
        self.desired_joint_values = trajectory[-1]
        self.trajectory_to_optimise = trajectory[1:-1]
        trajectory_flat = self.trajectory_to_optimise.T.reshape((-1))

        if withGrad:
            ################### With Gradient #########################
            now = time.time()

            opti_traj = opt.minimize(self.calculate_total_cost, trajectory_flat, method='BFGS',
                                     jac=self.cost_gradient_analytic, options={'maxiter': 150, 'disp': True},
                                     callback=self.optim_callback)

            cur = time.time()
            print 'CHOMP time with gradient:', cur - now

            opti_trajec = np.transpose(opti_traj.x.reshape((self.nDoF, -1)))
            opti_trajec = np.insert(opti_trajec, 0, self.initial_joint_values, axis=0)
            opti_trajec = np.insert(opti_trajec, len(opti_trajec), self.desired_joint_values, axis=0)
            # #
        else:
            # ###################### Without Gradient ############################
            now = time.time()
            opti_traj = opt.minimize(self.calculate_total_cost, trajectory_flat, method='BFGS',
                options={'maxiter': 150, 'disp': True}, callback=self.optim_callback)
            cur = time.time()
            print 'CHOMP time without gradient:', cur - now
            opti_trajec = np.transpose(opti_traj.x.reshape((self.nDoF, -1)))

            opti_trajec = np.insert(opti_trajec, 0, self.initial_joint_values, axis=0)
            opti_trajec = np.insert(opti_trajec, len(opti_trajec), self.desired_joint_values, axis=0)

        return opti_trajec, self.costs, opti_traj.success

    def calculate_total_cost(self, trajectoryFlat):
        F_smooth = self.smoothness_objective(trajectoryFlat)  # smoothness of trajectory is captured here
        if self.obstacle_list is not None:
            obstacle_cost = self.obstacle_cost(trajectoryFlat, self.obstacle_list)
        else:
            obstacle_cost = 0.0
        return self.lamda_smooth * F_smooth + self.lamda_obs * obstacle_cost  # for with gradient
        # return 10 * F_smooth + 1.5 * obstacle_cost   #  for without gradient

    def smoothness_objective(self, trajectoryFlat):
        trajectoryFlat = np.squeeze(trajectoryFlat)
        trajectory = trajectoryFlat.reshape((self.nDoF, -1)).T
        dim = np.multiply(*trajectory.shape)
        fd_matrix, b = self.finite_diff_matrix(trajectory)
        trajectory = trajectory.T.reshape((dim, 1))
        F_smooth = 0.5*(np.dot(trajectory.T, np.dot(fd_matrix, trajectory)) + np.dot(trajectory.T, b) + 0.25*np.dot(b.T, b))
        return F_smooth

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

    def obstacle_cost(self, trajectoryFlat, obstacle_list,):
        trajectoryFlat = np.squeeze(trajectoryFlat)
        trajectory = (trajectoryFlat.reshape((self.nDoF, trajectoryFlat.shape[0] / self.nDoF))).T
        robot_body_points = self.calculate_robot_body_points(trajectory)
        vel_normalised, vel_mag, vel = self.calculate_normalised_workspace_velocity(robot_body_points)
        dist = self.compute_minimum_distance_to_objects(robot_body_points[:-1], obstacle_list)
        obstacle_cost_potential = np.array(self.cost_potential(dist))
        return np.sum(np.multiply(obstacle_cost_potential, vel_mag))

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

    def compute_minimum_distance_to_objects(self, robot_body_points, obstacle_list, body_sphere_size=0.12):
        rbp_shape = robot_body_points.shape
        return np.min(
            distance.cdist(
                robot_body_points.reshape(-1, 3, order='A'),
                obstacle_list[:, :3]
            ) - obstacle_list[:, 3] - body_sphere_size,
            axis=1
        ).reshape(rbp_shape[:2])

    def calculate_smoothness_gradient(self, trajectoryFlat):
        trajectory = np.squeeze(trajectoryFlat).reshape((self.nDoF, -1)).T
        dim = np.multiply(*trajectory.shape)
        fd_matrix, b = self.finite_diff_matrix(trajectory)
        trajectory = trajectory.T.reshape((dim, 1))
        return 0.5*b + fd_matrix.dot(trajectory)

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
        trajectory = np.vstack((trajectory, self.desired_joint_values))
        return np.array([self.get_robot_discretised_points(self.get_robot_joint_positions(joint_values), self.step_size,
                                                           ) for joint_values in trajectory])

    def calculate_normalised_workspace_velocity(self, robot_body_points):
        velocity = np.diff(robot_body_points, axis=0)
        vel_magnitude = np.linalg.norm(velocity, axis=2)
        vel_normalised = np.divide(velocity, vel_magnitude[:, :, None], out=np.zeros_like(velocity), where=vel_magnitude[:, :, None] != 0)
        return vel_normalised, vel_magnitude, velocity

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

    def fun(self, points):
        dist = self.compute_minimum_distance_to_objects(points.reshape((1, 1, 3)), self.obstacle_list)
        return np.array(self.cost_potential(dist))

    def gradient_cost_potential(self, robot_discretised_points):
        gradient_cost_potential = np.zeros(robot_discretised_points.reshape(-1, 3).shape)
        for i, points in enumerate(robot_discretised_points.reshape((-1, 3))):
            gradient_cost_potential[i, :] = opt.approx_fprime(points, self.fun, [1e-06, 1e-06, 1e-06])
        return np.array(gradient_cost_potential).reshape(robot_discretised_points.shape[0], -1, 3)

    def calculate_jacobian2(self, robot_body_points, joint_index, trajectories=None):
        # Body Points: time X body points X 3
        # joint index: number of body points between joints as array
        # trajectory: time X number of joints
        joint_idxs = np.arange(self.nDoF)
        p_map = np.tril(np.ones((self.nDoF, self.nDoF), dtype=np.float32))
        t_joints = np.array(zip(*[self.franka_kin.fwd_kin(trajectory) for trajectory in trajectories])[1])
        joint_axis = t_joints[:, :, :3, 2]
        joint_positions = t_joints[:, :7, :3, -1]  # Excluding the fixed joint at the end
        jacobian = np.zeros((robot_body_points.shape[0], robot_body_points.shape[1], 3, self.nDoF))

        # s = robot_body_points.shape
        # robot_body_points = robot_body_points.reshape(-1, 3)
        for ti in range(trajectories.shape[0]):
            for i, points in enumerate(np.split(robot_body_points, joint_index)[:-1]):
                for point in points:
                    for joint in joint_idxs:
                        if i > 0 and p_map[i-1, joint]:
                            jacobian[ti, i, :, joint] = np.cross(joint_axis[ti, joint, :], point[ti]-joint_positions[ti, joint])
        return jacobian

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

    def calculate_obstacle_cost_gradient(self, trajectoryFlat):
        s = trajectoryFlat.shape
        trajectory = np.squeeze(trajectoryFlat).reshape((self.nDoF, -1)).T
        robot_discretised_points = self.calculate_robot_body_points(trajectory)
        vel_normalised, vel_magnitude, velocity = self.calculate_normalised_workspace_velocity(robot_discretised_points)
        curvature, orthogonal_projector = self.calculate_curvature(vel_normalised, vel_magnitude, velocity)
        dist = self.compute_minimum_distance_to_objects(robot_discretised_points[:-1], self.obstacle_list,)
        obstacle_cost_potential = np.array(self.cost_potential(dist))
        gradient_cost_potential = self.gradient_cost_potential(robot_discretised_points[:-1])
        obstacle_gradient = np.zeros((trajectory.shape[0], self.nDoF))
        # jacobian1 = self.calculate_jacobian2(robot_discretised_points[:-1], self.j_index, trajectory)
        a = robot_discretised_points[:-1]
        for jvs in range(trajectory.shape[0]):
            obst_grad = np.zeros(trajectory.shape[1])
            jacobian = self.calculate_jacobian(a[jvs], self.j_index, trajectory[jvs])
            for num_points in range(robot_discretised_points.shape[1]):
                temp1 = orthogonal_projector[jvs, num_points].dot(gradient_cost_potential[jvs, num_points, :])
                temp2 = obstacle_cost_potential[jvs, num_points] * curvature[jvs, num_points, :]
                temp3 = vel_magnitude[jvs, num_points] * (temp1 - temp2)
                obst_grad += jacobian[num_points, :].T.dot(temp3)
            obstacle_gradient[jvs, :] = obst_grad
        return obstacle_gradient.T

    def cost_gradient_analytic(self, trajectoryFlat):  # calculate grad(cost) = grad(smoothness_cost) + grad(obstacle_cost)
        smoothness_gradient = self.calculate_smoothness_gradient(trajectoryFlat)
        if self.obstacle_list is not None:
            obstacle_gradient = self.calculate_obstacle_cost_gradient(trajectoryFlat)
        else:
            obstacle_gradient = np.zeros(len(trajectoryFlat))
        trajectory = np.squeeze(trajectoryFlat)
        cost_gradient = self.lamda_obs * obstacle_gradient.reshape((len(trajectory), 1)) + self.lamda_smooth * smoothness_gradient
        return np.squeeze(cost_gradient)

    def cost_gradient_numeric(self, trajectory):
        trajectory = np.squeeze(trajectory)
        obst_cost_grad_numeric = opt.approx_fprime(trajectory, traj_opt.obstacle_cost,
                                              1e-08 * np.ones(len(trajectory)))
        smoothness_gradient = np.squeeze(self.calculate_smoothness_gradient(trajectory))
        return np.squeeze(obst_cost_grad_numeric + smoothness_gradient)

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
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.01)
            plt.pause(1)
        # plt.show(block=True)


# time index = 0; robot_discretised points index = 1, vector index = 2


if __name__ == '__main__':
    nDoF = 7
    # now = time.time()
    obstacle_list = np.array([[0.25, 0.21, 0.71, 0.1], [-0.15, 0.21, 0.51, 0.05]])
    obstacle_list1 = np.array([[0.15, -0.071, 0.79, 0.05], [-0.15, 0.21, 0.51, 0.05]])
    traj_opt_original = trajectory_optimization(nDoF)
    initial_joint_values = traj_opt_original.initial_joint_values
    desired_joint_values = traj_opt_original.desired_joint_values
    initial_trajectory = traj_opt_original.discretize(initial_joint_values, desired_joint_values, step_size=0.05)

    initial_trajectory = np.insert(initial_trajectory, len(initial_trajectory), desired_joint_values,
                                   axis=0)  # inserting goal state at the end of discretized
    initial_trajectory = np.insert(initial_trajectory, 0, initial_joint_values, axis=0)

    traj_opt = TrajectoryOptimize(nDoF, obstacle_list=obstacle_list)
    # traj_opt = TrajectoryOptimize(nDoF,)
    # obstacle_list = traj_opt.obstacle_list
    optimised_trajectory, _ = traj_opt.optimise(initial_trajectory,)
    traj_opt.animation(optimised_trajectory, initial_trajectory)
