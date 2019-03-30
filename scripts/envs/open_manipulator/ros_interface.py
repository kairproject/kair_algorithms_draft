# ! usr/bin/env python

import os

import numpy as np
from math import pi

import rospkg  # noqa
import rospy  # noqa
from gazebo_msgs.srv import DeleteModel, SpawnModel  # GetModelState
from geometry_msgs.msg import Point, Pose, Quaternion
from open_manipulator_msgs.msg import KinematicsPose, OpenManipulatorState
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from tf import TransformListener

# Global variables
TERM_COUNT = 10
SUC_COUNT = 10

base_dir = os.path.dirname(os.path.realpath(__file__))

overhead_orientation = Quaternion(
    x=-0.00142460053167, y=0.999994209902, z=-0.00177030764765, w=0.00253311793936
)


# safe joint limits
joint_limits = {
    'hi': {'j1': pi * 0.9, 'j2': pi * 0.5, 'j3': pi * 0.44, 'j4': pi * 0.65},
    'lo': {'j1': -pi * 0.9, 'j2': -pi * 0.57, 'j3': -pi * 0.3, 'j4': -pi * 0.57}}


class OpenManipulatorRosInterface:
    def __init__(self):
        rospy.init_node("OpenManipulatorRosInterface")

        self.tf = TransformListener()
        self.termination_count = 0
        self.success_count = 0

        self.joint_speeds = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.joint_positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.joint_velocities = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.joint_efforts = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.right_endpoint_position = [0, 0, 0]

        self.publisher_node()
        self.subscriber_node()
        self.init_robot_pose()

        rospy.on_shutdown(self._delete_target_block)

    def publish_node(self):
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

        self.joints_position_cmd = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.kinematics_cmd = [0.0, 0.0, 0.0]

    def subscribe_node(self):
        self.sub_joint_state = rospy.Subscriber(
            "/open_manipulator/joint_states", JointState, self.joint_state_callback
        )
        self.sub_kinematics_pose = rospy.Subscriber(
            "/open_manipulator/gripper/kinematics_pose",
            KinematicsPose,
            self.kinematics_pose_callback,
        )
        self.sub_robot_state = rospy.Subscriber(
            "/open_manipulator/states", OpenManipulatorState, self.robot_state_callback
        )

        self.joint_names = [
            "gripper",
            "gripper_sub",
            "joint1",
            "joint2",
            "joint3",
            "joint4",
        ]
        self.joint_positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.joint_velocities = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.joint_efforts = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.gripper_position = [0.0, 0.0, 0.0]
        self.gripper_orientiation = [0.0, 0.0, 0.0]
        self.distance_threshold = 0.1

        self.moving_state = ""
        self.actuator_state = ""

    def joint_state_callback(self, msg):
        """Callback function of joint states subscriber.

        Args:
            msg (JointState):  Callback message contains joint state.
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

        Args:
            msg (KinematicsPose): Callback message contains kinematics pose.
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

    def robot_state_callback(self, msg):
        """Callback function of robot state subscriber.

        Args:
            msg (states): Callback message contains openmanipulator's states.
        """
        # "MOVING" / "STOPPED"
        self.moving_state = msg.open_manipulator_moving_state
        # "ACTUATOR_ENABLE" / "ACTUATOR_DISABLE"
        self.actuator_state = msg.open_manipulator_actuator_state

    def get_joints_states(self):
        """Returns current joints states of robot including position,
        velocity, effort.

        Returns:
            Tuple of JointState
        """
        return self.joint_positions, self.joint_velocities, self.joint_efforts

    def get_gripper_pose(self):
        """Returns gripper end effector position.

        Returns:
            Tuple of Pose
        """
        return self.gripper_position, self.gripper_orientiation

    def get_gripper_position(self):
        """Returns gripper end effector position.

        Returns:
            Position
        """
        return self.gripper_position

    def set_joints_position(self, joints_angles):
        """Move joints using joint position command publishers.

        Args:
            joints_angles (Float64): Move joints with angles.
        """
        self.pub_gripper_position.publish(joints_angles[0])
        self.pub_gripper_sub_position.publish(joints_angles[1])
        self.pub_joint1_position.publish(joints_angles[2])
        self.pub_joint2_position.publish(joints_angles[3])
        self.pub_joint3_position.publish(joints_angles[4])
        self.pub_joint4_position.publish(joints_angles[5])

    def _reset_gazebo_world(self):
        """Initialize randomly the state of robot agent and
        surrounding envs (including target obj.).
        """
        self._delete_target_block()

        self.pub_gripper_position.publish(np.random.uniform(0.0, 0.1))
        self.pub_joint1_position.publish(np.random.uniform(-0.1, 0.1))
        self.pub_joint2_position.publish(np.random.uniform(-0.1, 0.1))
        self.pub_joint3_position.publish(np.random.uniform(-0.1, 0.1))
        self.pub_joint4_position.publish(np.random.uniform(-0.1, 0.1))
        self._load_target_block()

    def init_robot_pose(self):
        """Initialize robot girpper and joints position"""
        self.pub_gripper_position.publish(np.random.uniform(0.0, 0.1))
        self.pub_joint1_position.publish(np.random.uniform(-0.1, 0.1))
        self.pub_joint2_position.publish(np.random.uniform(-0.1, 0.1))
        self.pub_joint3_position.publish(np.random.uniform(-0.1, 0.1))
        self.pub_joint4_position.publish(np.random.uniform(-0.1, 0.1))
        self._load_target_block()

    def _load_target_block(self):
        """Load target block Gazebo model"""
        block_reference_frame = "world"
        model_path = rospkg.RosPack().get_path("kair_algorithms") + "/urdf/"

        block_xml = ""

        with open(model_path + "block/model.urdf", "r") as block_file:
            block_xml = block_file.read().replace("\n", "")

        rand_pose = Pose(
            position=Point(
                x=np.random.uniform(0.1, 0.15),
                y=np.random.uniform(0, 50.6),
                z=np.random.uniform(0.0, 0.1),
            ),
            orientation=overhead_orientation,
        )

        rospy.wait_for_service("/gazebo/spawn_urdf_model")

        try:
            spawn_urdf = rospy.ServiceProxy("/gazebo/spawn_urdf_model", SpawnModel)
            spawn_urdf("block", block_xml, "/", rand_pose, block_reference_frame)
        except rospy.ServiceException as e:
            rospy.logerr("Spawn URDF service call failed: {0}".format(e))

    def _delete_target_block(self):
        """This will be called on ROS Exit, deleting Gazebo models.

        Do not wait for the Gazebo Delete Model service, since
        Gazebo should already be running. If the service is not
        available since Gazebo has been killed, it is fine to error out
        """
        try:
            delete_model = rospy.ServiceProxy("/gazebo/delete_model", DeleteModel)
            delete_model("block")
        except rospy.ServiceException as e:
            rospy.loginfo("Delete Model service call failed: {0}".format(e))

    def _geom_interpolation(self, in_rad, out_rad, in_z, out_z, query):
        """interpolates along the outer shell of work space, based on z-position.

        must feed the corresponding radius from inner radius.
        """
        slope = (out_z - in_z) / (out_rad - in_rad)
        intercept = in_z
        return slope * (query - in_rad) + intercept

    def _check_for_success(self):
        """Check if the agent has succeeded the episode.

        Returns:
            True when count reaches SUC_COUNT, else False.
        """
        _dist = self._get_dist()
        if _dist < self.ros_interface.distance_threshold:
            self.success_count += 1
            if self.success_count == SUC_COUNT:
                self.done = True
                self.success_count = 0
                return True
            else:
                return False
        else:
            return False

    def _check_for_termination(self):
        """Check if the agent has reached undesirable state.

        If so, terminate the episode early.

        Returns:
            True when count reaches TERM_COUNT, else False.
        """
        _ee_pose = self.get_gripper_position()

        inner_rad = 0.134
        outer_rad = 0.3
        lower_rad = 0.384
        inner_z = 0.321
        outer_z = 0.250
        lower_z = 0.116
        rob_rad = np.linalg.norm([_ee_pose[0], _ee_pose[1]])
        rob_z = _ee_pose[2]
        if self.joint_positions[0] <= abs(joint_limits['hi']['j1'] / 2):
            if rob_rad < inner_rad:
                self.termination_count += 1
                rospy.logwarn('OUT OF BOUNDARY : exceeds inner radius limit')
            elif inner_rad <= rob_rad < outer_rad:
                upper_z = self._geom_interpolation(inner_rad, outer_rad,
                                                   inner_z, outer_z,
                                                   rob_rad)
                if rob_z > upper_z:
                    self.termination_count += 1
                    rospy.logwarn('OUT OF BOUNDARY : exceeds upper z limit')
            elif outer_rad <= rob_rad < lower_rad:
                bevel_z = self._geom_interpolation(outer_rad, lower_rad,
                                                   outer_z, lower_z,
                                                   rob_rad)
                if rob_z > bevel_z:
                    self.termination_count += 1
                    rospy.logwarn('OUT OF BOUNDARY : exceeds bevel z limit')
            else:
                self.termination_count += 1
                rospy.logwarn('OUT OF BOUNDARY : exceeds outer radius limit')
        else:
            # joint_1 limit exceeds
            self.termination_count += 1
            rospy.logwarn('OUT OF BOUNDARY : joint_1_limit exceeds')

        if self.termination_count == TERM_COUNT:
            self.done = True
            self.termination_count = 0
            return True
        else:
            return False

    def close(self):
        """Close by rospy shutdown."""
        rospy.signal_shutdown("done")
