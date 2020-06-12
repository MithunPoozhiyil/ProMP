#!/usr/bin/env python

import os
# import tf
import csv
import rospy
import yaml
import numpy as np
import time, sys
import subprocess
import moveit_msgs.msg
import moveit_commander
from pprint import pprint as pp
from geometry_msgs.msg import PoseStamped
from chomp import TrajectoryOptimize
from traj_opt_weight_space import optim_weight_space
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.msg import RobotState, RobotTrajectory
from franka_kinematics import FrankaKinematics
import tf.transformations as tf_tran
from scipy.interpolate import CubicSpline
# from whycon.msg import detection_results_array


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print '%r  %2.2f ms (%2.2f fps)' % (method.__name__, (te - ts) * 1000, 1. / (te - ts))
        return result

    return timed


class Move():

    def __init__(self, *args):
        moveit_commander.roscpp_initialize(sys.argv)
        self.scene = moveit_commander.PlanningSceneInterface()
        self.setup_planner()
        self.display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                            moveit_msgs.msg.DisplayTrajectory, queue_size=20)
        self.workspace_dim = (-0.5, -0.5, 0.2, 1.5, 0.5, 1.9)

        self.pub_pose = rospy.Subscriber('/bottle/centroid', PoseStamped, self.callback)

    def callback(self, msg):
        self.mu_x = [getattr(msg.pose.position, i) for i in ['x', 'y', 'z']]

    def discretize(self, start, goal, step_size=0.02):
        w = goal - start
        step = step_size * w / np.linalg.norm(w)
        n = int(np.linalg.norm(w) / np.linalg.norm(step)) + 1
        discretized = np.outer(np.arange(1, n), step) + start
        return discretized

    def setup_planner(self):
        self.group = moveit_commander.MoveGroupCommander("panda_arm")
        # self.group.set_workspace((-1, -0.5, 0, 1, 0.5, 2))  # specify the workspace bounding box (static scene)
        # self.group.set_workspace((-0.1,-0.05,0,0.1,0.05,0.2))

        # self.group.set_end_effector_link("panda_hand")    # planning wrt to panda_hand or link8
        self.group.set_max_velocity_scaling_factor(0.15)  # scaling down velocity
        self.group.set_max_acceleration_scaling_factor(0.05)  # scaling down velocity
        self.group.allow_replanning(True)
        self.group.set_num_planning_attempts(5)
        self.group.set_goal_position_tolerance(0.05)
        self.group.set_goal_orientation_tolerance(0.03)
        self.group.set_planning_time(30)
        # self.group.set_planner_id("BiTRRTkConfigDefault")
        # self.group.set_planner_id("FMTkConfigDefault")
        # self.group.set_planner_id("RRTConnectkConfigDefault")
        # self.group.set_planner_id("CHOMP")
        print "CURRENT JOINT VALUES:", self.group.get_current_joint_values()
        self.group.set_start_state_to_current_state()

    def populate_obstacles(self, ):
        # if the obstacle is within robot workspace, then consider those as obstacles
        obs = []
        x_min, x_max = self.workspace_dim[0], self.workspace_dim[3]
        y_min, y_max = self.workspace_dim[1], self.workspace_dim[4]
        z_min, z_max = self.workspace_dim[2], self.workspace_dim[5]

        for k, v in self.scene.get_objects().items():
            px, py, pz = v.primitive_poses[0].position.x, v.primitive_poses[0].position.y, v.primitive_poses[
                0].position.z
            if px >= x_min and px <= x_max:
                if py >= y_min and py <= y_max:
                    if pz >= z_min and pz <= z_max:
                        obs.append([px, py, pz, 0.10])
        return np.array(obs)

    def terminate_ros_node(self, s):
        # Adapted from http://answers.ros.org/question/10714/start-and-stop-rosbag-within-a-python-script/
        list_cmd = subprocess.Popen("rosnode list", shell=True, stdout=subprocess.PIPE)
        list_output = list_cmd.stdout.read()
        retcode = list_cmd.wait()
        assert retcode == 0, "List command returned %d" % retcode
        for str in list_output.split("\n"):
            if (str.startswith(s)):
                os.system("rosnode kill " + str)

    def go_to_joint_state(self, joint_goal):
        self.group.go(joint_goal, wait=True)
        self.group.stop()

    def plan(self, X, Y, Z, qx, qy, qz, qw):
        pose_target = PoseStamped()
        pose_target.header.frame_id = self.group.get_planning_frame()

        pose_target.pose.orientation.x = qx  # was 0.924 for link8
        pose_target.pose.orientation.y = qy
        pose_target.pose.orientation.z = qz
        pose_target.pose.orientation.w = qw

        pose_target.pose.position.x = X
        pose_target.pose.position.y = Y
        pose_target.pose.position.z = Z

        pp(pose_target)
        self.group.set_pose_target(pose_target)

        start = time.time()
        plan = self.group.plan()
        total_time = time.time() - start
        return plan, total_time

    def calculate_stats(self, plan, plan_time, obstacle_list, smth_cost_promp=0.0):
        results = dict()
        joint_positions = plan
        if not isinstance(plan, (list, tuple, np.ndarray)):
            n = len(plan.joint_trajectory.points)
            joint_positions = np.zeros((n, 7))
            for i in range(n):
                joint_positions[i, :] = plan.joint_trajectory.points[i].positions

        traj_opt = TrajectoryOptimize(7, obstacle_list, joint_positions[0, :], joint_positions[-1, :])
        joint_positions_flat = joint_positions.T.reshape(-1)
        smoothness_cost = traj_opt.smoothness_objective(joint_positions_flat)
        if obstacle_list is not None:
            obstacle_cost = traj_opt.obstacle_cost(joint_positions_flat, obstacle_list)
        else:
            obstacle_cost = 0.0
        path_length = np.sum(np.linalg.norm(np.diff(joint_positions, axis=0), axis=1))
        results['plan_time (s)'] = plan_time
        results['path_length'] = path_length
        results['smoothness_cost_as_in_chomp'] = smoothness_cost[0][0]
        if smth_cost_promp:
            results['smoothness_cost_promp'] = smth_cost_promp[0][0]
        else:
            results['smoothness_cost_promp'] = 0.0
        results['obstacle_cost'] = obstacle_cost
        return results

    def plan_execute(self, X, Y, Z, qx, qy, qz, qw):
        plan, plan_time = self.plan(X, Y, Z, qx, qy, qz, qw)

        print " Waiting while RVIZ displays plan..."
        rospy.sleep(1)

        while True:
            text = raw_input("============ Press Y to execute and N to terminate")
            if text == "Y" or text == "y":
                break
            if text == "N" or text == "n":
                self.group.clear_pose_targets()
                self.group.stop()
                raise ValueError('User wanted me to quit :(')
        print "Executing"
        self.group.execute(plan)

        return

    def create_q_v_a(self, q):  # q is of dimension n x nDoF
        n, nDoF = q.shape
        vel_scale_fact, accel_scale_fact = 0.8, 0.35
        tc = np.ones(2)
        for i in range(nDoF):
            _, _, tme = self.time_scaling(q[:, i], vel_scale_fact, accel_scale_fact, )
            if tme[-1] > tc[-1]:
                tc = tme
        vel, accel = np.zeros((len(tc), nDoF)), np.zeros((len(tc), nDoF))

        for i in range(nDoF):
            cs = CubicSpline(tc, q[:, i], bc_type='clamped')
            vel[:, i], accel[:, i] = cs(tc, 1), cs(tc, 2)

        return vel, accel, tc

    def time_scaling(self, q_list, vel_scale_fact, accel_scale_fact, ):
        vel_limit, accel_limit = 2.1, 1.8
        vel_permisible, accel_permisible = vel_limit * vel_scale_fact, accel_limit * accel_scale_fact
        tf = 2
        t = np.linspace(0, tf, len(q_list))
        dt = t[1] - t[0]
        cs = CubicSpline(t, q_list, bc_type='clamped')
        vel, accel = cs(t, 1), cs(t, 2)
        vel_max, accel_max = np.max(abs(vel)), np.max(abs(accel))
        while vel_max > vel_permisible or accel_max > accel_permisible:
            tf += dt
            t = np.linspace(0, tf, len(q_list))
            dt = t[1] - t[0]
            cs = CubicSpline(t, q_list, bc_type='clamped')
            vel, accel = cs(t, 1), cs(t, 2)
            vel_max, accel_max = np.max(abs(vel)), np.max(abs(accel))
        return vel, accel, t

    def create_trajectory(self, q_list, v_list, a_list, t):
        tnsec, tsec = np.modf(t)
        jt = JointTrajectory()
        jt.header.frame_id = '/panda_link0'
        jt.points = []
        for i, tm in enumerate(t):
            p = JointTrajectoryPoint()
            p.positions = q_list[i, :]
            p.velocities = v_list[i, :]
            p.accelerations = a_list[i, :]
            p.time_from_start.secs = tsec[i]
            p.time_from_start.nsecs = tnsec[i] * 1e09
            jt.points.append(p)
        jt.joint_names = ["panda_joint1",
                          "panda_joint2",
                          "panda_joint3",
                          "panda_joint4",
                          "panda_joint5",
                          "panda_joint6",
                          "panda_joint7"]
        rt = RobotTrajectory()
        rt.joint_trajectory = jt
        return rt  # (rt for move_group; jt for motion_planning)

    def promp_planner(self, mu_x, sig_x, mu_quat, sig_quat, obstacle_list, Qlists, timeLists, PromP_Planner=True):
        time_normalised = np.linspace(0, 1, 35)
        nDoF, nBf = 7, 5
        cur_jv = self.group.get_current_joint_values()
        weight_space_optim = optim_weight_space(time_normalised, nDoF, nBf, mu_x, sig_x, mu_quat, sig_quat,
                                                obstacle_list, curr_jvs=cur_jv, QLists=Qlists, timeLists=timeLists)
        weights = np.random.multivariate_normal(weight_space_optim.mean_cond_weight,
                                                weight_space_optim.covMat_cond_weight)

        if not PromP_Planner:
            ################ Trajectory space optimization: CHOMP

            franka_kin = FrankaKinematics()
            des_jv = weight_space_optim.post_mean_q
            n = np.linalg.norm((des_jv - cur_jv)) / (len(time_normalised) - 1)

            traj2optimise = self.discretize(cur_jv, des_jv, step_size=n)
            traj2optimise = np.insert(traj2optimise, len(traj2optimise), des_jv,
                                           axis=0)  # inserting goal state at the end of discretized
            traj2optimise = np.insert(traj2optimise, 0, cur_jv, axis=0)
            print 'chomp n', n, traj2optimise.shape[0]
            save_path = '/home/automato/Ash/MPlib/src/ProMPlib/with_obstacles/CHOMP/'
            uniq_name = str(time.time())

            traj_opt = TrajectoryOptimize(7, obstacle_list)
            start = time.time()
            optimised_trajectory = traj_opt.optimise(traj2optimise, withGrad=True)
            np.savetxt(save_path + 'traj_CHOMP_%s.out' % uniq_name, optimised_trajectory, delimiter=',')
            total_time = time.time() - start
            print total_time
            T, _ = franka_kin.fwd_kin(optimised_trajectory[-1])
            quat = np.array(tf_tran.quaternion_from_matrix(T))
            print 'quat_diff = ', quat - mu_quat
            print '#########'
            print 'pos_diff = ', T[:3, 3] - mu_x

            vel, accel, tme = self.create_q_v_a(optimised_trajectory)
            rt = self.create_trajectory(optimised_trajectory, vel, accel, tme)
            rt_back = self.create_trajectory(np.flipud(optimised_trajectory), np.flipud(vel), np.flipud(accel), tme)
            smooth_cost_promp = 0.0
            return optimised_trajectory, total_time, rt, rt_back, smooth_cost_promp

        ##################################
        else:
            # Weight space optimization:

            start = time.time()
            optimised_trajectory, opt_weights = weight_space_optim.optimise(weights, withGrad=True)
            smooth_cost_promp = weight_space_optim.smoothness_cost(opt_weights)
            total_time = time.time() - start
            print total_time
            save_path = '/home/automato/Ash/MPlib/src/ProMPlib/with_obstacles/ProMP/'
            uniq_name = str(time.time())

            np.savetxt(save_path+'traj_promp_%s.out' % uniq_name, optimised_trajectory, delimiter=',')
            np.savetxt(save_path+'weights_promp_%s.out' % uniq_name, opt_weights, delimiter=',')

            franka_kin = FrankaKinematics()
            T, _ = franka_kin.fwd_kin(optimised_trajectory[-1])
            quat = np.array(tf_tran.quaternion_from_matrix(T))
            print 'quat_diff = ', quat - mu_quat
            print '#########'
            print 'pos_diff = ', T[:3, 3] - mu_x
            vel, accel, tme = self.create_q_v_a(optimised_trajectory)
            rt = self.create_trajectory(optimised_trajectory, vel, accel, tme)
            rt_back = self.create_trajectory(np.flipud(optimised_trajectory), np.flipud(vel), np.flipud(accel), tme)

            return optimised_trajectory, total_time, rt, rt_back, smooth_cost_promp


if __name__ == "__main__":

    with open('/home/automato/Ash/MPlib/src/trajecV2B.npz',
              'r') as f:
        data = np.load(f)
        Q1 = data['Q']
        timeData1 = data['time']

    with open('/home/automato/Ash/MPlib/src/bottle2cup_first_trial.npz',
              'r') as f:
        data = np.load(f)
        Q2 = data['Q']
        timeData2 = data['time']

    A1 = []
    for i in range(len(Q1)):  # Q has 9 joints including 2 gripper joints, so making it 7
        A1.append(Q1[i][:, :-2])

    A2 = []
    for i in range(len(Q2)):  # Q has 9 joints including 2 gripper joints, so making it 7
        A2.append(Q2[i][:, :-2])

    rospy.init_node('planner_tester', anonymous=True)

    joint_start_prompTC = [-0.2321, -0.314 , -0.2275, -0.3768,  0.0411,  1.1934,  0.6216]
    move_robot = Move()
    time.sleep(2)

    # mu_xbottle = move_robot.mu_x
    mu_xV2B = np.loadtxt('positionSampleV2B.txt', dtype=float)
    QuaternionV2B = np.loadtxt('QuartSampleV2B.txt', dtype=float)
    mu_xB2C = np.loadtxt('positionSampleB2C.txt', dtype=float)
    QuaternionB2C = np.loadtxt('QuartSampleB2C.txt', dtype=float)

    stats = dict()
    try:

        obstacle_list = np.array(move_robot.populate_obstacles())
        print obstacle_list.shape
        if not obstacle_list.size:
            obstacle_list = None
        move_robot.go_to_joint_state(joint_start_prompTC)
        planner_ids = ["RRT", "PRMkConfigDefault", "CHOMP", "ProMP"]

        unique_name = str(time.time())
        pth = '/home/automato/Ash/MPlib/src/ProMPlib/with_obstacles/PRM/'
        with open(pth + "PRM_%s.csv" % unique_name, 'w') as csvfile:
            fieldnames = ['planner', 'plan_time (s)', 'path_length', 'smoothness_cost_as_in_chomp', 'smoothness_cost_promp', 'obstacle_cost']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            while not rospy.is_shutdown():
                planner_index = int(raw_input(
                    "enter 0: RRTConnectkConfigDefault, 1: PRMkConfigDefault, 2: CHOMP, "
                    "3: ProMP\n").strip())

                idxx = int(raw_input('Enter sample mu_x and Quart: 0 - 10\n').strip())

                data2learnFrom = int(raw_input('Enter index for motion:V2B = 1, B2C = 2\n').strip())
                if data2learnFrom == 1:
                    Qlists = A1
                    timeLists = timeData1
                elif data2learnFrom == 2:
                    Qlists = A2
                    timeLists = timeData2
                else:
                    data2learnFrom = 0

                idx = data2learnFrom

                if planner_index == 0 or planner_index == 1:
                    planner_id = planner_ids[planner_index]
                    move_robot.group.set_planner_id(planner_id)
                    if idx == 0:
                        raw_input('Move to vertical?')
                        move_robot.go_to_joint_state(joint_start_prompTC)

                    elif idx == 1:
                        plan, plan_time = move_robot.plan(
                            mu_xV2B[idxx][0], mu_xV2B[idxx][1], mu_xV2B[idxx][2],
                            QuaternionV2B[idxx][0], QuaternionV2B[idxx][1], QuaternionV2B[idxx][2], QuaternionV2B[idxx][3]
                        )
                        save_path = '/home/automato/Ash/MPlib/src/ProMPlib/with_obstacles/PRM/'
                        uniq_name = str(time.time())
                        with open(save_path + "traj_PRM_%s.yaml" %uniq_name, 'w') as outfile:
                            yaml.dump(plan, outfile, default_flow_style=False)

                        stats = move_robot.calculate_stats(plan, plan_time, obstacle_list)
                        raw_input('Move?')
                        move_robot.group.execute(plan)
                    elif idx == 2:
                        plan, plan_time = move_robot.plan(
                            mu_xB2C[idxx][0], mu_xB2C[idxx][1], mu_xB2C[idxx][2],
                            QuaternionB2C[idxx][0], QuaternionB2C[idxx][1], QuaternionB2C[idxx][2], QuaternionB2C[idxx][3]
                        )
                        save_path = '/home/automato/Ash/MPlib/src/ProMPlib/with_obstacles/PRM/'
                        uniq_name = str(time.time())
                        with open(save_path + "traj_PRM_%s.yaml" % uniq_name, 'w') as outfile:
                            yaml.dump(plan, outfile, default_flow_style=False)
                        stats = move_robot.calculate_stats(plan, plan_time, obstacle_list)
                        raw_input('Move?')
                        move_robot.group.execute(plan)
                    print planner_id
                    stats['planner'] = planner_id
                    writer.writerow(stats)
                    rospy.sleep(1)
                    print 'RRT:', move_robot.group.get_current_joint_values()
                    # move_robot.go_to_joint_state(joint_start_prompTC)

                elif planner_index == 2 or planner_index == 3:
                    if planner_index == 3:
                        ProMP = True
                    else:
                        ProMP = False

                    planner_id = planner_ids[planner_index]
                    if obstacle_list is not None:
                        print obstacle_list.shape
                    if data2learnFrom == 1:
                        plan, plan_time, rt, rt_back, scp = move_robot.promp_planner(mu_xV2B[idxx], np.eye(3) * 1e-04,
                                                                                QuaternionV2B[idxx],
                                                                                np.eye(4) * 1e-05, obstacle_list,
                                                                                Qlists, timeLists, PromP_Planner=ProMP)
                        stats = move_robot.calculate_stats(plan, plan_time, obstacle_list, smth_cost_promp=scp)
                    else:
                        plan, plan_time, rt, rt_back, scp = move_robot.promp_planner(mu_xB2C[idxx], np.eye(3) * 1e-04,
                                                                                QuaternionB2C[idxx],
                                                                                np.eye(4) * 1e-05, obstacle_list,
                                                                                Qlists, timeLists, PromP_Planner=ProMP)
                        stats = move_robot.calculate_stats(plan, plan_time, obstacle_list, smth_cost_promp=scp)
                    print planner_id
                    stats['planner'] = planner_id
                    writer.writerow(stats)
                    while(raw_input('Enter g') != "g"):
                        pass
                    move_robot.group.execute(rt)
                    rospy.sleep(1)
                    print move_robot.group.get_current_joint_values()
                    # raw_input('Enter to go back')
                    # move_robot.group.execute(rt_back)

                    # move_robot.go_to_joint_state(joint_start_prompTC)

        # while True:
        #     pose = int(raw_input(
        #         "enter 0: exit, 1: original pose and 2: goal pose, 3: joint_home, 4: joint_random, 5: To tomato").strip())
        #
        #     if pose == 0:
        #         break
        #     elif pose == 1:
        #         Run_Moveit.plan_execute(-0.031, -0.066, 1.029, -0.46, 0.4271, -0.559, 0.5414)
        #     elif pose == 2:
        #         Run_Moveit.plan_execute(0.4, -0.10, 0.75, -0.528, 0.543, -0.478, 0.446)
        #     elif pose == 3:
        #         # Run_Moveit.record_rosbag = True
        #         # print "starting record"
        #         # Run_Moveit.go([-1.93, -1.758, 1.02, -1.84, 2.351, 2.4, 1.076])
        #         Run_Moveit.go([0.01646, -0.0127182, -0.02679, -0.09966, 0.0221, 0.11921, 0.012])
        #         # print "ending record"
        #         # Run_Moveit.record_rosbag = False
        #     elif pose == 4:
        #         # Run_Moveit.record_rosbag = True
        #         # print "starting record"
        #         Run_Moveit.go([-2.755, -1.537, 1.056, -1.913, 2.861, 1.527, 0.557])
        #         # print "ending record"
        #         # Run_Moveit.record_rosbag = False
        #     elif pose == 5:
        #         Run_Moveit.plan_execute(transPLT[0], transPLT[1], transPLT[2], qrtPLT[0], qrtPLT[1], qrtPLT[2],
        #                                 qrtPLT[3])
        #     else:
        #         continue
    except rospy.ROSInterruptException:
        print("program interupted before completion")
