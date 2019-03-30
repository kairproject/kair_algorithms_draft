#! /usr/bin/env python

import copy
import math
import os
import sys
import time
from random import *
from string import Template
from math import pi, cos, sin, radians
import cv2
import numpy as np
import rospkg

import rospy
from control_msgs.msg import JointTrajectoryControllerState
from cv_bridge import CvBridge, CvBridgeError
from gazebo_msgs.msg import ContactsState
from gazebo_msgs.srv import DeleteModel, GetModelState, SetModelState, SpawnModel
# reads open_manipulator's state
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from open_manipulator_msgs.msg import *
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import *
from tf import TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

base_dir = os.path.dirname(os.path.realpath(__file__))

overhead_orientation = Quaternion(
    x=-0.00142460053167, y=0.999994209902, z=-0.00177030764765, w=0.00253311793936
)

# safe joint limits
joint_limits = {'hi':{'j1':pi*0.9,  'j2':pi*0.5,   'j3':pi*0.44, 'j4': pi*0.65 },
                'lo':{'j1':-pi*0.9, 'j2':-pi*0.57, 'j3':-pi*0.3, 'j4':-pi*0.57 }}

cartesian_limits = {}
# safe cartesian limits 
# robot @ home-pose : (0.134, 0.0, 0.241)

# episode termination condition
X_MIN = 0.1
X_MAX = 0.5
Y_MIN = -0.3
Y_MAX = 0.3
Z_MIN = 0.0
Z_MAX = 0.6
TERM_COUNT = 10
SUC_COUNT = 10


# Global variables
# -------------------------
ACTION_DIM = 3  # Cartesian
OBS_DIM = (100, 100, 3)  # POMDP
STATE_DIM = 24  # MDP


class OpenManipulatorEnv:
    def __init__(
        self,
        max_steps=700,
        isdagger=False,
        isPOMDP=False,
        isreal=False,
        train_indicator=0,
    ):
        """An implementation of OpenAI-Gym style robot reacher environment
        TODO: add method that receives target object's pose as state
        """
        rospy.init_node("OpenManipulatorEnv")
        self.train_indicator = train_indicator  # 0: Train 1:Test
        self.isdagger = isdagger
        self.isPOMDP = isPOMDP
        self.isreal = isreal
        self.currentDist = 1
        self.previousDist = 1
        self.reached = False
        self.tf = TransformListener()
        self.bridge = CvBridge()
        self.joint_speeds = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.joint_positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.joint_velocities = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.joint_efforts = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.right_endpoint_position = [0, 0, 0]
        self.max_steps = max_steps
        self.done = False
        self.reward = 0
        self.reward_rescale = 1.0
        self.isDemo = False
        self.reward_type = "sparse"
        self.termination_count = 0
        self.success_count = 0

        self.pub_gripper_position = rospy.Publisher(
            "/open_manipulator/gripper_position/command", Float64, queue_size=1
        )
        self.pub_gripper_sub_position = rospy.Publisher(
            "/open_manipulator/gripper_sub_position/command", Float64, queue_size=1
        )
        self.pub_joint1_position = rospy.Publisher(
            "/open_manipulator/joint1_position/command", Float64, queue_size=1
        )
        self.pub_joint2_position = rospy.Publisher(
            "/open_manipulator/joint2_position/command", Float64, queue_size=1
        )
        self.pub_joint3_position = rospy.Publisher(
            "/open_manipulator/joint3_position/command", Float64, queue_size=1
        )
        self.pub_joint4_position = rospy.Publisher(
            "/open_manipulator/joint4_position/command", Float64, queue_size=1
        )

        # TODO: manage this attribute when it's real test environment
        self.joints_position_cmd = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.kinematics_cmd = [0.0, 0.0, 0.0]
        self.sub_joint_state = rospy.Subscriber(
            "/open_manipulator/joint_states", JointState, self.joint_state_callback
        )
        # joint position/velocity/effort
        self.sub_kinematics_pose = rospy.Subscriber(
            "/open_manipulator/gripper/kinematics_pose",
            KinematicsPose,
            self.kinematics_pose_callback,
        )
        self.sub_robot_state = rospy.Subscriber(
            "/open_manipulator/states", OpenManipulatorState, self.robot_state_callback
        )
        # cs position / orientation
        # variables for subscribe the joint states
        self.joint_names = [
            "gripper",
            "gripper_sub",
            "joint1",
            "joint2",
            "joint3",
            "joint4",
        ]  # name: [gripper, gripper_sub, joint1, joint2, joint3, joint4]
        self.joint_positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.joint_velocities = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.joint_efforts = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # ee pose of robot -> used to compute reward.
        self.gripper_position = [0.0, 0.0, 0.0]  # [x, y, z] cartesian position
        self.gripper_orientiation = [
            0.0,
            0.0,
            0.0,
        ]  # [x, y, z, w] quaternion orientation
        self.distance_threshold = 0.1

        # used for per-step elapsed time measurement
        self.tic = 0.0
        self.toc = 0.0
        self.elapsed = 0.0
        # self.initial_state = self.get_joints_states().copy()
        self._action_scale = 1.0

        # open manipulator statets
        self.moving_state = ""
        self.actuator_state = ""
        self.init_robot_pose()
        rospy.on_shutdown(self._delete_target_block)

    
    def render(self):
        pass
    
    def robot_state_callback(self, msg):
        self.moving_state = msg.open_manipulator_moving_state  # "MOVING" / "STOPPED"
        self.actuator_state = (
            msg.open_manipulator_actuator_state
        )  # "ACTUATOR_ENABLE" / "ACTUATOR_DISABLE"

    def joint_state_callback(self, msg):
        """Callback function of joint states subscriber.
        Argument: msg
        """
        self.joints_states = msg
        self.joint_names = self.joints_states.name
        self.joint_positions = self.joints_states.position
        self.joint_velocities = self.joints_states.velocity
        self.joint_efforts = self.joints_states.effort
        # penalize jerky motion in reward for shaped reward setting.
        self.squared_sum_vel = np.linalg.norm(np.array(self.joint_velocities))

    def kinematics_pose_callback(self, msg):
        """Callback function of gripper kinematic pose subscriber.
        Argument: msg
        """
        self.kinematics_pose = msg
        _gripper_position = self.kinematics_pose.pose.position
        self.gripper_position = [
            _gripper_position.x,
            _gripper_position.y,
            _gripper_position.z,
        ]
        _gripper_orientiation = self.kinematics_pose.pose.orientation
        self.gripper_orientiation = [
            _gripper_orientiation.x,
            _gripper_orientiation.y,
            _gripper_orientiation.z,
            _gripper_orientiation.w,
        ]

    # get and set function
    # ----------------------------

    def get_joints_states(self):
        """Returns current joints states of robot including position, velocity, effort
        Returns: Float64[] self.joints_position, self.joints_velocity, self.joint_effort
        """
        return self.joint_positions, self.joint_velocities, self.joint_efforts

    def get_gripper_pose(self):
        """Returns gripper end effector position
        Returns: Pose().position, Pose().orientation
        """
        return self.gripper_position, self.gripper_orientiation

    def get_gripper_position(self):
        """Returns gripper end effector position
        Returns: Pose().position
        """
        return self.gripper_position

    def set_joints_position(self, joints_angles):
        """Move joints using joint position command publishers.
        Argument: joints_position_cmd
        self.joints_position_cmd = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        """
        # rospy.loginfo(Set joint position)
        self.pub_gripper_position.publish(joints_angles[0])
        self.pub_gripper_sub_position.publish(joints_angles[1])
        self.pub_joint1_position.publish(joints_angles[2])
        self.pub_joint2_position.publish(joints_angles[3])
        self.pub_joint3_position.publish(joints_angles[4])
        self.pub_joint4_position.publish(joints_angles[5])

    def step(self, action=np.array([1, 1, 1, 1, 1, 1]), step=0):
        """Function executed each time step.
        Here we get the action execute it in a time step and retrieve the
        observations generated by that action.
        :param action:
        :return: obs, reward, done
        """
        self.prev_tic = self.tic
        self.tic = time.time()
        self.elapsed = time.time() - self.prev_tic
        self.done = False
        if step == self.max_steps:
            self.done = True
        act = action.flatten().tolist()
        self.set_joints_position(act)
        curDist = self._get_dist()
        if not self.isreal:
            self.reward = self._compute_reward()
            if self._check_for_termination():
                print("======================================================")
                print("Terminates current Episode : OUT OF BOUNDARY")
                print("======================================================")
            elif self._check_for_success():
                print("======================================================")
                print("Succeeded current Episode")
                print("======================================================")
        _joint_pos, _joint_vels, _joint_effos = self.get_joints_states()
        # obj_pos = self._get_target_obj_obs() # TODO: implement this function call.

        if np.mod(step, 10) == 0:
            if not self.isreal:
                print("DISTANCE : ", curDist)
            print("PER STEP ELAPSED : ", self.elapsed)
            print("SPARSE REWARD : ", self.reward_rescale * self.reward)
            print("Current EE pos: ", self.gripper_position)
            print("Actions: ", act)

        obs = np.array([_joint_pos, _joint_vels, _joint_effos])
        info = ''

        return obs, self.reward_rescale * self.reward, self.done, info

    def reset(self):
        # Attempt to reset the simulator. Since we randomize initial conditions, it
        # is possible to get into a state with numerical issues (e.g. due to penetration or
        # Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        # In this case, we just keep randomizing until we eventually achieve a valid initial
        # configuration.
        did_reset_sim = False
        self._reset_gazebo_world()
        _joint_pos, _joint_vels, _joint_effos = self.get_joints_states()
        obs = np.array([_joint_pos, _joint_vels, _joint_effos])

        return obs

    def _check_robot_moving(self):
        """Check if robot has reached its initial pose.
        """
        while not rospy.is_shutdown():
            if self.moving_state == "STOPPED":
                break
        return True

    def _reset_gazebo_world(self):
        """
        Method that randomly initialize the state of robot agent and surrounding envs (including target obj.)
        """
        self._delete_target_block()

        self.pub_gripper_position.publish(np.random.uniform(0.0, 0.1))
        self.pub_joint1_position.publish(np.random.uniform(-0.1, 0.1))
        self.pub_joint2_position.publish(np.random.uniform(-0.1, 0.1))
        self.pub_joint3_position.publish(np.random.uniform(-0.1, 0.1))
        self.pub_joint4_position.publish(np.random.uniform(-0.1, 0.1))
        self._load_target_block()

    def init_robot_pose(self):
        self.pub_gripper_position.publish(np.random.uniform(0.0, 0.1))
        self.pub_joint1_position.publish(np.random.uniform(-0.1, 0.1))
        self.pub_joint2_position.publish(np.random.uniform(-0.1, 0.1))
        self.pub_joint3_position.publish(np.random.uniform(-0.1, 0.1))
        self.pub_joint4_position.publish(np.random.uniform(-0.1, 0.1))
        self._load_target_block()

    def _delete_target_block(self):
        # This will be called on ROS Exit, deleting Gazebo models
        # Do not wait for the Gazebo Delete Model service, since
        # Gazebo should already be running. If the service is not
        # available since Gazebo has been killed, it is fine to error out
        try:
            delete_model = rospy.ServiceProxy("/gazebo/delete_model", DeleteModel)
            resp_delete = delete_model("block")
        except rospy.ServiceException as e:
            rospy.loginfo("Delete Model service call failed: {0}".format(e))

    def _load_target_block(self,
        block_pose=Pose(position=Point(x=0.6725, y=0.1265, z=0.7825)),
        block_reference_frame="world",
    ):
        # Get Models' Path
        model_path = rospkg.RosPack().get_path("kair_algorithms") + "/urdf/"

        # Load Block URDF
        block_xml = ""

        with open(model_path + "block/model.urdf", "r") as block_file:
            block_xml = block_file.read().replace("\n", "")

        # Spawn Block URDF
        rospy.wait_for_service("/gazebo/spawn_urdf_model")
        try:
            spawn_urdf = rospy.ServiceProxy("/gazebo/spawn_urdf_model", SpawnModel)
            resp_urdf = spawn_urdf(
                "block", block_xml, "/", block_pose, block_reference_frame
            )
        except rospy.ServiceException as e:
            rospy.logerr("Spawn URDF service call failed: {0}".format(e))

    def _geom_interpolation(self, in_rad, out_rad, in_z, out_z, query):
        """interpolates along the outer shell of work space, based on z-position.
        must feed the corresponding radius from inner radius.
        """
        slope = (out_z - in_z)/(out_rad - in_rad)
        intercept = in_z
        return slope*(query - in_rad) + intercept                                                                                              


    def _check_for_termination(self):
        """
        Check if the agent has reached undesirable state. If so, terminate the episode early.
        based on the polar coordinate
        """
        _ee_pose = self.get_gripper_position()
        # define gemetry
        inner_rad = 0.134
        outer_rad = 0.3
        lower_rad = 0.384
        inner_z = 0.321
        outer_z = 0.250
        lower_z = 0.116
        rob_rad = np.linalg.norm([_ee_pose[0], _ee_pose[1]])
        rob_z = _ee_pose[2]
        if self.joint_positions[0] <= abs(joint_limits['hi']['j1']/2):
            if rob_rad < inner_rad:
                self.termination_count += 1
                rospy.logwarn('OUT OF BOUNDARY : exceeds inner radius limit')
            elif inner_rad <= rob_rad < outer_rad:
                upper_z = self._geom_interpolation(inner_rad, outer_rad, inner_z, outer_z, rob_rad)
                if rob_z > upper_z:
                    self.termination_count += 1
                    rospy.logwarn('OUT OF BOUNDARY : exceeds upper z limit')
            elif outer_rad <= rob_rad < lower_rad:
                bevel_z = self._geom_interpolation(outer_rad, lower_rad, outer_z, lower_z, rob_rad)
                if rob_z > bevel_z:
                    self.termination_count += 1
                    rospy.logwarn('OUT OF BOUNDARY : exceeds bevel z limit')
            else:
                self.termination_count += 1
                rospy.logwarn('OUT OF BOUNDARY : exceeds outer radius limit')
        else: # joint_1 limit exceeds
            self.termination_count += 1
            rospy.logwarn('OUT OF BOUNDARY : joint_1_limit exceeds')

        if self.termination_count == TERM_COUNT:
            self.done = True
            self.termination_count = 0
            return True
        else:
            return False

    def _check_for_success(self):
        """
        Check if the agent has succeeded the episode.
        """
        _dist = self._get_dist()
        if _dist < self.distance_threshold:
            self.success_count += 1
            if self.success_count == SUC_COUNT:
                self.done = True
                self.success_count = 0
                return True
        else:
            return False

    def _compute_reward(self):
        """Computes shaped/sparse reward for each episode.
        """
        cur_dist = self._get_dist()
        if self.reward_type == "sparse":
            return (cur_dist <= self.distance_threshold).astype(
                np.float32
            )  # 1 for success else 0
        else:
            return -cur_dist - self.squared_sum_vel  # -L2 distance -l2_norm(joint_vels)

    def _get_dist(self):
        rospy.wait_for_service("/gazebo/get_model_state")
        try:
            object_state_srv = rospy.ServiceProxy(
                "/gazebo/get_model_state", GetModelState
            )
            object_state = object_state_srv("block", "world")
            self._obj_pose = np.array(
                [
                    object_state.pose.position.x,
                    object_state.pose.position.y,
                    object_state.pose.position.z,
                ]
            )
        except rospy.ServiceException as e:
            rospy.logerr("Spawn URDF service call failed: {0}".format(e))
        _ee_pose = np.array(self.get_gripper_position())  # FK state of robot
        return np.linalg.norm(_ee_pose - self._obj_pose)

    def close(self):
        rospy.signal_shutdown("done")
