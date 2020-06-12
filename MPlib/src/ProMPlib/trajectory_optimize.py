#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import rospy
import time
# from cost_function import CostFunction
import matplotlib.pyplot as plt
import moveit_commander
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import PoseStamped

from mpl_toolkits.mplot3d import Axes3D
import scipy.ndimage
from franka_kinematics import FrankaKinematics
import scipy.optimize as opt


class trajectory_optimization():

    def __init__(self, nDoF):
        self.nDoF = nDoF
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group = moveit_commander.MoveGroupCommander("panda_arm")
        self.robot_state = moveit_commander.RobotCommander()
        self.workspace_dim = (-2, -0.2, 0.2, 1, 1, 1.5)
        self.group.set_workspace(self.workspace_dim)  # specify the workspace bounding box (static scene)
        self.franka_kin = FrankaKinematics()
        self.collision_threshold = 0.07
        self.object_list = np.array([[0.25, 0.21, 0.71, 0.1],
                                     [-0.15, 0.21, 0.51, 0.05],
                                     [ 0.15, 0.61, 0.51, 0.05],
                                     [-0.15, -0.61, 0.51, 0.05],
                                     [-0.65, 0.31, 0.21, 0.05],
                                     [-0.55, 0.71, 0.71, 0.06],
                                     [-0.95, -0.51, 0.41, 0.07],
                                     [-0.85, 0.27, 0.61, 0.08],
                                     [-0.25, -0.31, 0.11, 0.04],
                                     [-0.75, 0.71, 0.91, 0.09],
                                     [-0.35, 0.51, 0.81, 0.02],
                                     [-0.95, -0.81, 0.61, 0.03],
                                     [-0.75, 0.11, 0.41, 0.02],
                                     [-0.55, -0.61, 0.21, 0.06],
                                     [ 0.25, 0.31, 0.71, 0.07],
                                     [-0.35, -0.41, 0.41, 0.02],
                                     [-0.25, -0.51, 0.61, 0.08],
                                     ])  # for two spheres
        # self.object_list = np.array([[0.25, 0.21, 0.71, 0.1], [-0.15, 0.21, 0.51, 0.05]])  # for two spheres
        # self.object_list = np.array([[0.25, 0.21, 0.71, 0.1]])  # for one sphere (x,y,z, radius)
        # self.object_list = []
        # self.object_list = self.populate_obstacles()
        self.initial_joint_values = np.zeros(7)  # + 0.01 * np.random.random(7)
        self.ang_deg = 60
        self.desired_joint_values = np.array(
            [np.pi * self.ang_deg / 180, np.pi / 3, 0.0, np.pi / 6, np.pi / 6, np.pi / 6, np.pi / 6])
        self.optCurve = []
        self.step_size = 0.09

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
            # a.append([px, py, pz, 0.05])

        return obs
            # print "Ori:", v.primitive_poses[0].orientation.x, v.primitive_poses[0].orientation.y, \
            #     v.primitive_poses[0].orientation.z, v.primitive_poses[0].orientation.w
            # print "Size:", v.primitives[0].dimensions

            # self.object_list.append([v.primitive_poses[0].position.x, v.primitive_poses[0].position.y, \
            #     v.primitive_poses[0].position.z, 0.05])

    def discretize(self, start, goal, step_size=0.02):
        w = goal - start
        step = step_size * w / np.linalg.norm(w)
        n = int(np.linalg.norm(w) / np.linalg.norm(step)) + 1
        discretized = np.outer(np.arange(1, n), step) + start
        return discretized

    def get_robot_discretised_points(self, joint_values=None, step_size=0.2, with_joint_index=False):

        if joint_values is None:
            joint_values = list(self.robot_state.get_current_state().joint_state.position)[:7]

        _, t_joints = self.franka_kin.fwd_kin(joint_values)
        fwd_k_j_positions = np.vstack(([0.,0.,0.], t_joints[:, :3, 3]))
        discretised = list()
        j_index = list()
        for j in range(len(fwd_k_j_positions) - 1):
            w = fwd_k_j_positions[j+1] - fwd_k_j_positions[j]
            if len(w[w != 0.]):
                step = step_size * w / np.linalg.norm(w)
                n = int(np.linalg.norm(w) / np.linalg.norm(step)) + 1
                discretised.extend(np.outer(np.arange(1, n), step) + fwd_k_j_positions[j])
            j_index.append(len(discretised))
        return (np.array(discretised), j_index) if with_joint_index else np.array(discretised)

    def compute_minimum_distance_to_objects(self, robot_body_points, object_list, ):
        # if object_list is None:
        #     object_list = self.get_object_collision_spheres()
        D = list()
        if len(robot_body_points.shape) == 1:
            robot_body_points = robot_body_points.reshape(1, 3)

        for r in robot_body_points:  # expects robot_body_points as an array of dimension 1 x n
            dist = []
            for o in object_list:
                ro = r - o[:3]
                norm_ro = np.linalg.norm(ro)
                dist.append(norm_ro - 0.15 - o[3])
            D.append(np.min(np.array(dist)))
        return D

    def cost_potential(self, D):
        c = list()
        for d in D:
            if d < 0.:
                c.append(-d + 0.5 * self.collision_threshold)
            elif d <= self.collision_threshold:
                c.append((0.5 * (d-self.collision_threshold)**2) / self.collision_threshold)
            else:
                c.append(0)
        return c

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

    def smoothness_objective(self, trajectory):
        trajectoryFlat = np.squeeze(trajectory)
        trajectory = (trajectoryFlat.reshape((self.nDoF, trajectoryFlat.shape[0] / self.nDoF))).T
        rows, columns = trajectory.shape
        dim = rows * columns
        fd_matrix, b = self.finite_diff_matrix(trajectory)
        trajectory = trajectory.T.reshape((dim, 1))
        F_smooth = 0.5*(np.dot(trajectory.T, np.dot(fd_matrix, trajectory)) + np.dot(trajectory.T, b) + 0.25*np.dot(b.T, b))
        return F_smooth

    def calculate_obstacle_cost(self, trajectory):
        obstacle_cost = 0
        trajectoryFlat = np.squeeze(trajectory)
        trajectory = (trajectoryFlat.reshape((self.nDoF, trajectoryFlat.shape[0] / self.nDoF))).T
        vel_normalised, vel_mag, vel = self.calculate_normalised_workspace_velocity(
            trajectory)  # vel_normalised = vel/vel_mag
        for jvs in range(len(trajectory)):
            # print trajectory.shape

            robot_discretised_points = np.array(
                self.get_robot_discretised_points(trajectory[jvs], self.step_size))
            dist = self.compute_minimum_distance_to_objects(robot_discretised_points, self.object_list)
            obsacle_cost_potential = np.array(self.cost_potential(dist))
            obstacle_cost += np.sum(np.multiply(obsacle_cost_potential, vel_mag[jvs, :]))
            # obstacle_cost += np.sum(obsacle_cost_potential)
        return obstacle_cost

    def calculate_total_cost(self, trajectory):
        trajectory = np.squeeze(trajectory)
        F_smooth = self.smoothness_objective(trajectory)  # smoothness of trajectory is captured here
        obstacle_cost = self.calculate_obstacle_cost(trajectory)
        return 10 * F_smooth + 25.5 * obstacle_cost     #  for with gradient
        # return 10 * F_smooth + 1.5 * obstacle_cost   #  for without gradient

    def calculate_normalised_workspace_velocity(self, trajectory, desired_joint_values=None):
        if desired_joint_values is None:
            desired_joint_values = self.desired_joint_values

        # We have not divided by  time as this has been indexed and is thus not available
        trajectory = np.insert(trajectory, len(trajectory), desired_joint_values, axis=0)
        robot_body_points = np.array([self.get_robot_discretised_points(joint_values, self.step_size) for joint_values in trajectory])
        velocity = np.diff(robot_body_points, axis=0)
        vel_magnitude = np.linalg.norm(velocity, axis=2)
        vel_normalised = np.divide(velocity, vel_magnitude[:, :, None], out=np.zeros_like(velocity), where=vel_magnitude[:, :, None] != 0)
        return vel_normalised, vel_magnitude, velocity

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

    def calculate_smoothness_gradient(self, trajectory):
        trajectoryFlat = np.squeeze(trajectory)
        trajectory = (trajectoryFlat.reshape((self.nDoF, trajectoryFlat.shape[0] / self.nDoF))).T
        rows, columns = trajectory.shape
        dim = rows * columns
        fd_matrix, b = self.finite_diff_matrix(trajectory)
        trajectory = trajectory.T.reshape((dim, 1))
        smoothness_gradient = 0.5*b + fd_matrix.dot(trajectory)
        return smoothness_gradient

    def calculate_curvature(self, vel_normalised, vel_magnitude, velocity):
        time_instants, body_points, n = velocity.shape[0], velocity.shape[1], velocity.shape[2]
        acceleration = np.gradient(velocity, axis=0)
        curvature, orthogonal_projector = np.zeros((time_instants, body_points, n)), np.zeros((time_instants, body_points, n, n))
        for tm in range(time_instants):
            for pts in range(body_points):
                ttm = np.dot(vel_normalised[tm, pts, :].reshape(3, 1), vel_normalised[tm, pts, :].reshape(1, 3))
                temp = np.eye(3) - ttm
                orthogonal_projector[tm, pts] = temp
                if vel_magnitude[tm, pts]:
                    curvature[tm, pts] = np.dot(temp, acceleration[tm, pts, :])/vel_magnitude[tm, pts]**2
                else:
                    # curv.append(np.array([0, 0, 0]))
                    curvature[tm, pts] = np.array([0, 0, 0])
        return curvature, orthogonal_projector

    def fun(self, points):
        dist = self.compute_minimum_distance_to_objects(points, self.object_list)
        return np.array(self.cost_potential(dist))

    def gradient_cost_potential(self, robot_discretised_points):
        gradient_cost_potential = list()
        for points in robot_discretised_points:
            grad = opt.approx_fprime(points, self.fun, [1e-06, 1e-06, 1e-06])
            gradient_cost_potential.append(grad)
        return np.array(gradient_cost_potential)

    def calculate_mt(self, trajectory):
        n_time, ndof = trajectory.shape  #
        M_t =np.zeros((n_time, ndof, n_time*ndof))
        for t in range(n_time):
            k = 0
            for d in range(ndof):
                M_t[t, d, k + t] = 1
                k = k + n_time
        return M_t

    def calculate_obstacle_cost_gradient(self, trajectory):
        trajectoryFlat = np.squeeze(trajectory)
        trajectory = (trajectoryFlat.reshape((self.nDoF, trajectoryFlat.shape[0] / self.nDoF))).T
        vel_normalised, vel_magnitude, velocity = self.calculate_normalised_workspace_velocity(trajectory)
        curvature, orthogonal_projector = self.calculate_curvature(vel_normalised, vel_magnitude, velocity)
        obstacle_gradient = list()
        # a, b, c, d = [], [], [], []
        for jvs in range(len(trajectory)):
            obst_grad = np.zeros(trajectory.shape[1])
            robot_discretised_points, joint_index = np.array(self.get_robot_discretised_points(trajectory[jvs], self.step_size, with_joint_index=True))
            dist = np.array(self.compute_minimum_distance_to_objects(robot_discretised_points, self.object_list,))
            obstacle_cost_potential = np.array(self.cost_potential(dist))
            gradient_cost_potential = self.gradient_cost_potential(robot_discretised_points)
            jacobian = self.calculate_jacobian(robot_discretised_points, joint_index, trajectory[jvs])
            # a.append(robot_discretised_points)
            # b.append(obstacle_cost_potential)
            # c.append(gradient_cost_potential)
            # d.append(jacobian)
            for num_points in range(robot_discretised_points.shape[0]):
                temp1 = orthogonal_projector[jvs, num_points].dot(gradient_cost_potential[num_points, :])
                temp2 = obstacle_cost_potential[num_points] * curvature[jvs, num_points, :]
                temp3 = vel_magnitude[jvs, num_points] * (temp1 - temp2)
                obst_grad += jacobian[num_points, :].T.dot(temp3)
                # obst_grad += jacobian[num_points, :].T.dot(gradient_cost_potential[num_points, :])
            obstacle_gradient.append(obst_grad)
        return np.transpose(np.array(obstacle_gradient))

    def cost_gradient_analytic(self, trajectory):  # calculate grad(cost) = grad(smoothness_cost) + grad(obstacle_cost)
        smoothness_gradient = self.calculate_smoothness_gradient(trajectory)
        obstacle_gradient = self.calculate_obstacle_cost_gradient(trajectory)
        trajectory = np.squeeze(trajectory)
        cost_gradient = obstacle_gradient.reshape((len(trajectory), 1)) + smoothness_gradient
        return np.squeeze(cost_gradient)

    def cost_gradient_numeric(self, trajectory):
        trajectory = np.squeeze(trajectory)
        obst_cost_grad_numeric = opt.approx_fprime(trajectory, traj_opt.calculate_obstacle_cost,
                                                   1e-08 * np.ones(len(trajectory)))
        smoothness_gradient = np.squeeze(self.calculate_smoothness_gradient(trajectory))
        return np.squeeze(obst_cost_grad_numeric + smoothness_gradient)

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
                for object in self.object_list:
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

    def optim_callback(self, xk):
        # xk = xk.reshape(len(xk) / 7, 7)
        costs = self.calculate_total_cost(xk)
        self.optCurve.append(xk)
        print 'Iteration {}: {}\n'.format(len(self.optCurve), costs)

# time index = 0; robot_discretised points index = 1, vector index = 2


if __name__ == '__main__':
    nDoF = 7
    traj_opt = trajectory_optimization(nDoF)

    initial_joint_values = traj_opt.initial_joint_values
    desired_joint_values = traj_opt.desired_joint_values
    trajectory = traj_opt.discretize(initial_joint_values, desired_joint_values, step_size=0.3)

    initial_trajectory = trajectory
    initial_trajectory = np.insert(initial_trajectory, len(initial_trajectory), desired_joint_values, axis=0)  # inserting goal state at the end of discretized
    initial_trajectory = np.insert(initial_trajectory, 0, initial_joint_values, axis=0)  # inserting start state at the start of discretized

    trajectoryFlat = trajectory.T.reshape((-1))

    # smoothness_grad_analytic = np.squeeze(traj_opt.calculate_smoothness_gradient(trajectoryFlat))
    # smoothness_grad_numeric = opt.approx_fprime(trajectoryFlat, traj_opt.smoothness_objective, 1e-08*np.ones(len(trajectoryFlat)))

    # obst_cost_grad_analytic = np.reshape(traj_opt.calculate_obstacle_cost_gradient(trajectoryFlat), -1)
    # obst_cost_grad_numeric = opt.approx_fprime(trajectoryFlat, traj_opt.calculate_obstacle_cost,
    #                                             1e-08 * np.ones(len(trajectoryFlat)))

    # obst_cost_grad_numeric_test = opt.approx_fprime(trajectoryFlat, traj_opt.calculate_obstacle_cost_test,
    #                                            1e-08 * np.ones(len(trajectoryFlat)))

    # print obst_cost_grad_numeric_test

    # smoothness_grad_numeric1 = np.zeros(trajectoryFlat.shape)
    # for i in range(trajectoryFlat.shape[0]):
    #     trajectoryTemp = trajectoryFlat.copy()
    #     trajectoryTemp[i] = trajectoryTemp[i] + 10**-8
    #     val1 = traj_opt.smoothness_objective(trajectoryTemp)
    #     trajectoryTemp[i] = trajectoryTemp[i] - 2 * 10 ** -8
    #     val2 = traj_opt.smoothness_objective(trajectoryTemp)
    #     smoothness_grad_numeric1[i] = (val1 - val2) / (2 * 10 ** -8)

    ################### With Gradient ############################
    now = time.time()

    optimized_trajectory = opt.minimize(traj_opt.calculate_total_cost, trajectoryFlat, method='BFGS', jac=traj_opt.cost_gradient_analytic, options={'maxiter': 30, 'disp': True}) # , callback=traj_opt.optim_callback)

    cur = time.time()
    print cur - now
    #
    # optimized_trajectory = np.transpose(optimized_trajectory.x.reshape((7, len(optimized_trajectory.x) / 7)))
    # optimized_trajectory = np.insert(optimized_trajectory, 0, initial_joint_values, axis=0)
    # optimized_trajectory = np.insert(optimized_trajectory, len(optimized_trajectory), desired_joint_values, axis=0)
    # print 'result \n', optimized_trajectory
    # traj_opt.animation(optimized_trajectory, initial_trajectory)


    ############ Without Gradient ##########################
    # optimized_trajectory = opt.minimize(traj_opt.calculate_total_cost, trajectoryFlat, method='BFGS',
    #              options={'maxiter': 10, 'disp': True}, callback=traj_opt.optim_callback)
    # jac = optimized_trajectory.jac
    # # total_cost_gradient = traj_opt.cost_gradient_numeric(optimized_trajectory.x)
    # smoothness_gradient = np.squeeze(traj_opt.calculate_smoothness_gradient(optimized_trajectory.x))
    # obstacl_cost_grad_numeric = opt.approx_fprime(optimized_trajectory.x, traj_opt.calculate_obstacle_cost,
    #                                            1e-08 * np.ones(len(optimized_trajectory.x)))
    # print jac - (smoothness_gradient + obstacl_cost_grad_numeric)
    #
    # optimized_trajectory = np.transpose(optimized_trajectory.x.reshape((7, len(optimized_trajectory.x) / 7)))
    # optimized_trajectory = np.insert(optimized_trajectory, 0, initial_joint_values, axis=0)
    # optimized_trajectory = np.insert(optimized_trajectory, len(optimized_trajectory), desired_joint_values, axis=0)
    # print 'result \n', optimized_trajectory
    # traj_opt.animation(optimized_trajectory, initial_trajectory)

    # print 'finished'







