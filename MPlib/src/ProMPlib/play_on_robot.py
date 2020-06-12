#!/usr/bin/env python

import numpy as np
from threading import Lock
import sys
import moveit_commander
from moveit_msgs.msg import RobotState, RobotTrajectory
from moveit_msgs.srv import GetStateValidity, GetStateValidityRequest
import rospy
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint, MultiDOFJointTrajectory
from pprint import pprint as pp


# with open('/home/ash/Ash/Repo/MPlib/src/ProMPlib/traject_task_conditioned_infront.npz', 'r') as f:
#     trajectories = np.load(f)

with open('/home/ash/Ash/Repo/MPlib/src/ProMPlib/traject_task_conditioned_very_good.npz', 'r') as f:
    trajectories = np.load(f)

class MoveArm(object):

    def __init__(self):
        print "Motion Planning Initializing..."
        # Prepare the mutex for synchronization
        self.mutex = Lock()


        # Some info and conventions about the robot that we hard-code in here
        # min and max joint values are not read in Python urdf, so we must hard-code them here
        self.q_list = trajectories
        self.num_joints = 7
        self.frame_id = '/panda_link0'
        self.joint_names = ["panda_joint1",
                            "panda_joint2",
                            "panda_joint3",
                            "panda_joint4",
                            "panda_joint5",
                            "panda_joint6",
                            "panda_joint7"]

        # Publish trajectory command
        self.pub_trajectory = rospy.Publisher("/joint_trajectory", JointTrajectory,
                                              queue_size=1)        

        # Initialize variables
        self.joint_state = JointState()
        print self.joint_state

        # Wait for validity check service
        rospy.wait_for_service("check_state_validity")
        self.state_valid_service = rospy.ServiceProxy('check_state_validity', GetStateValidity)
        print "State validity service ready"

        # Initialize MoveIt
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group_name = "panda_arm"
        self.group = moveit_commander.MoveGroupCommander(self.group_name)

        joint_goal = self.group.get_current_joint_values()
        joint_goal = [-0.01943573, -0.021066, -0.01686345, -0.09939517, -0.00450579,  0.12849557,
  0.78490839]
        print "rotating joint ...."
        self.group.go(joint_goal, wait=True)

        print "MoveIt! interface ready"

        # Options
        self.subsample_trajectory = True
        print "Initialization done."

        ###############################################################

    def set_joint_val(self, joint_state, q, name):
        if name not in joint_state.name:
            print "ERROR: joint name not found"
        i = joint_state.name.index(name)
        joint_state.position[i] = q

    """ Given a complete joint_state data structure, this function finds the values for 
    our arm's set of joints in a particular order and returns a list q[] containing just 
    those values.
    """

    def joint_state_from_q(self, joint_state, q):
        for i in range(0, self.num_joints):
            self.set_joint_val(joint_state, q[i], self.joint_names[i])

    """ This function checks if a set of joint angles q[] creates a valid state, or 
    one that is free of collisions. The values in q[] are assumed to be values for 
    the joints of the left arm, ordered from proximal to distal. 
    """
    def is_state_valid(self, q):
        req = GetStateValidityRequest()
        req.group_name = self.group_name
        # current_joint_state = deepcopy(self.joint_state)
        # current_joint_state.position = q
        # current_joint_state.name = self.joint_names
        # self.joint_state_from_q(current_joint_state, q)
        req.robot_state = RobotState()
        req.robot_state.joint_state.position = q
        req.robot_state.joint_state.name = self.joint_names
        res = self.state_valid_service(req)
        return res.valid

    ##################################################

    # def create_trajectory(self, q_list, v_list, a_list, t):
    #     joint_trajectory = JointTrajectory()
    #     for i in range(0, len(q_list)):
    #         point = JointTrajectoryPoint()
    #         point.positions = list(q_list[i])
    #         point.velocities = list(v_list[i])
    #         point.accelerations = list(a_list[i])
    #         point.time_from_start = rospy.Duration(t[i])
    #         joint_trajectory.points.append(point)
    #     joint_trajectory.joint_names = self.joint_names
    #     return joint_trajectory

    def create_trajectory(self, q_list):
        jt = JointTrajectory()
        jt.header.frame_id = self.frame_id
        jt.points = []
        for i in range(0, len(q_list)):
            p = JointTrajectoryPoint()
            p.positions = q_list[i, :]
            p.time_from_start.secs = i/10.
            p.time_from_start.nsecs = i/10. + 0.002
            jt.points.append(p)
        jt.joint_names = self.joint_names
        rt = RobotTrajectory()
        rt.joint_trajectory = jt
        return rt # (rt for move_group; jt for motion_planning)
        
    def joint_states_callback(self, joint_state):
        self.mutex.acquire()
        self.joint_state = joint_state
        self.mutex.release()

    def execute(self, joint_trajectory):
        self.pub_trajectory.publish(joint_trajectory)


if __name__ == '__main__':
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('play_on_robot', anonymous=True)
    ma = MoveArm()
    rospy.sleep(0.5)

    #####################################
    #  Play all trajectories on the robot

    for k in range(trajectories.shape[2] - 15):
        asa = ma.q_list[:, :, k]
        # For execution with MoveIt (This cant play all the sampled trajectories)
        # joint_trajectory = ma.create_trajectory(asa)
        # ma.group.execute(joint_trajectory)
        # joint_trajectory_flipped = ma.create_trajectory(np.flipud(asa))
        # ma.group.execute(joint_trajectory_flipped)

        # For execution with motion_planning package
        joint_trajectory = ma.create_trajectory(asa)
        ma.execute(joint_trajectory)
        rospy.sleep(4)
        joint_trajectory_flipped = ma.create_trajectory(np.flipud(asa))
        ma.execute(joint_trajectory)
        rospy.sleep(4)
        raw_input('here')

    ###################################################
    # To find out collision free trajectory and play on robot

    for i in range(0, trajectories.shape[2]):
        chk_list = []
        for j in range(1, trajectories.shape[0]):
            q = trajectories[j, :, i]
            chk_list.append(ma.is_state_valid(q))
        if all(chk_list):
            coll_free_traj = trajectories[:, :, i]
            # print i
            break
    joint_trajectory = ma.create_trajectory(coll_free_traj)
    raw_input('Enter to play collision free trajectory')
    ma.execute(joint_trajectory)
    rospy.sleep(4)
    joint_trajectory_flipped = ma.create_trajectory(np.flipud(coll_free_traj))
    ma.execute(joint_trajectory_flipped)
    rospy.sleep(4)
    print 'finished'
