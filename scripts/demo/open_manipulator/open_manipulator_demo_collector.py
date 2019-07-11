#!/usr/bin/env python
"""Collect demos using jacobian based control.

- Author: DH Kim
- Contact: kdh0429@snu.ac.kr
"""
import json
import random
import threading
import time
from collections import defaultdict
from math import pi, pow
import numpy as np
import rospy
from pykdl_utils.kdl_kinematics import KDLKinematics
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Float64MultiArray
from urdf_parser_py.urdf import URDF


class DemoCollector(object):
    """Demo collector class which controls openmanipulator based on jacobain method."""

    def __init__(self):
        rospy.loginfo("Start Demo Collector")
        # TODO: Receive True or False with parser to check real or simulation.
        self.use_platform = rospy.get_param("~use_platform", False)
        self.print_start_message()

        self.robot_urdf = URDF.from_parameter_server()
        self.robot = KDLKinematics(self.robot_urdf, "world", "end_effector_link")
        self.save_path = "../DemoCollection.json"

        self.init_shared_variables()
        self.init_observation()

        self.init_subscriber()
        self.init_publisher()

    def print_start_message(self):
        if self.use_platform is False:
            rospy.loginfo("Start Gazebo Demo Collector")
        else:
            rospy.loginfo("Start Real Demo Collector")

    def init_shared_variables(self):
        """Initialize shared variables.

        q, qdot, effort: 4 size array.
        Receive from joint state callback.

        q: [joint1_position, joint2_position, joint3_position, joint4_position]
        qdot: [joint1_velocity, joint2_velocity, joint3_velocity, joint4_velocity]
        effot: [joint1_torque, joint2_torque, joint3_torque, joint4_torque]
        ex) [q1, q2, q3, q4]

        T: 4x4 transformation matrix
        Get matrix by solve forward kinematics.
        [R p
         0 1]
          - R: Rotation matrix
          - p: position
        ex)
        [s -c  0  x
         c  s  0  y
         0  0  1  z
         0  0  0  1]

        [Homogeneous Transformation Matrices]
          - material: https://www.youtube.com/watch?v=vlb3P7arbkU
        [Forward Kinematics]
          - material: https://www.youtube.com/watch?v=hE_Duih_7JE&list=PLggLP4f-rq00efLcgMcG1m4k5CKlgRcGh
        """
        self.mutex = threading.Lock()
        self.damping = rospy.get_param("~damping", 0.01)
        self.joint_vel_limit = rospy.get_param("~joint_vel_limit", 4)
        self.q = np.zeros(4)  # Joint angles
        self.q_desired = np.zeros(4)
        self.qdot = np.zeros(4)  # Joint velocities
        self.effort = np.zeros(4)  # Joint torque
        self.T_target = np.array(self.robot.forward(self.q))
        self.T_goal = np.array(self.robot.forward(self.q))
        self.T_cur = np.array(self.robot.forward(self.q))

        # TODO: extract num_tar_demo to config
        self.num_tar_demo = 3
        self.num_cur_demo = 0

        self.control_start_time = self.get_rostime()

    def init_observation(self):
        """Initialize observation"""
        self._gripper_pos = np.zeros(3)
        self._gripper_orientation = np.zeros(4)

    def init_subscriber(self):
        """Initialize joint states subscriber."""
        if self.use_platform is False:
            self.joint_states_sub = rospy.Subscriber(
                "/open_manipulator/joint_states", JointState, self.joint_states_cb
            )
        else:
            self.joint_states_sub = rospy.Subscriber(
                "/open_manipulator/joint_states_real", JointState, self.joint_states_cb
            )

    def init_publisher(self):
        """Initialize joint command publisher."""
        self.j1_pos_command_pub = rospy.Publisher(
            "/open_manipulator/joint1_position/command", Float64, queue_size=3
        )
        self.j2_pos_command_pub = rospy.Publisher(
            "/open_manipulator/joint2_position/command", Float64, queue_size=3
        )
        self.j3_pos_command_pub = rospy.Publisher(
            "/open_manipulator/joint3_position/command", Float64, queue_size=3
        )
        self.j4_pos_command_pub = rospy.Publisher(
            "/open_manipulator/joint4_position/command", Float64, queue_size=3
        )
        self.joint_pos_command_to_dxl_pub = rospy.Publisher(
            "/open_manipulator/joint_position/command", Float64MultiArray, queue_size=3
        )

    def joint_states_cb(self, joint_states):
        """ Save joint states published in ROS to class member."""
        for i in range(4):
            self.q[i] = joint_states.position[i + 2]
            self.qdot[i] = joint_states.velocity[i + 2]
            self.effort[i] = joint_states.effort[i + 2]

    def get_rostime(self):
        return rospy.get_rostime().secs + rospy.get_rostime().nsecs * 10 ** -9

    def run(self):
        """Run demo collection.

        1) If init is false, do initial setting.
        2) If target is not set, and robot is initial pose, set new target.
        3) If target is set, move to target.
        4) If robot pose is not initial pose, move to initial pose.
        5) If target is set and robot pose is not initial pose, take ros sleep.
        6) If number of target demo is bigger than number of current demo, finish demo
        collection.
        """
        # TODO: extract 100 to config
        self.hz = 100
        self.r = rospy.Rate(self.hz)
        self.start_log()
        self.q_init = list(self.q)
        self.done_move_to_target = False

        for i in range(self.num_tar_demo):
            print("Episode: ", i)
            rospy.loginfo("Moving to Initial Position")
            # go to init pose
            self.done_init = False
            self.control_start_time = self.get_rostime()
            while not self.done_init:
                self.move_to_init()
                self.r.sleep()

            self.set_target()
            # TODO: replace is_set_new_target to done(move_to_target)
            self.T_init = np.array(self.robot.forward(self.q))
            # go to target
            self.control_start_time = self.get_rostime()
            while not self.done_move_to_target:
                self.move_to_target(i)  # run 3
                self.r.sleep()

        print("Demo Collection Finished!")
        self.save_demo_collection(self.save_path)
        quit()

    def save_demo_collection(self, save_path):
        with open(save_path, "w") as f:
            json.dump(self.data, f)
        print("Demo file saved successfully")

    def start_log(self):
        """ Start logging in dict(dict(list)) type."""
        self.data = defaultdict(lambda: defaultdict(list))

    def set_target(self):
        """ Randomly set target within joint limit and workspace limit.

        q_limit_L: Low limit of joint.
        q_limit_H: High limit of joint.
        q_rand: Randomly set joint angle.
        T target: Transformation matrix of randomly set target.
                  Get matrix by robot forward kinematics of q_rand.
        target: Position of randomly set target.
                Get target array from transformation matrix.
        min_op_distance: Minimum workspace distance limit.
        max_op_distance: Maximum workspace distance limit.

        1) Initialize and set new target if target is not appropriate.
        2) Get random joint angle within joint limits.
        3) Get transformation matrix and position array by robot forward kinmetics.
        4) Calculate target position within workspace distance limits.
        5) Check if it is appropriate target. Finish loop if it is true, othercase
           continue loop.
        """
        appropriate_target = False
        while appropriate_target is False:
            q_limit_L = [-pi * 0.5, -pi * 0.5, -pi * 0.3, -pi * 0.57]
            q_limit_H = [pi * 0.5, pi * 0.5, pi * 0.44, pi * 0.65]
            rand_scale = np.zeros(4)
            q_rand = np.zeros(4)
            for i in range(4):
                rand_scale[i] = random.random()
                q_rand[i] = rand_scale[i] * (q_limit_H[i] - q_limit_L[i]) + q_limit_L[i]

            self.T_target = np.array(self.robot.forward(q_rand))
            target = np.empty_like(self.T_target[:3, 3])
            target[:] = self.T_target[:3, 3]

            min_op_distance = 0.15
            max_op_distance = 0.4

            if np.linalg.norm(np.abs(target)) > max_op_distance:
                target = target * max_op_distance / np.linalg.norm(np.abs(target))

            if np.linalg.norm(np.abs(target)) < min_op_distance:
                target = target * min_op_distance / np.linalg.norm(np.abs(target))

            if target[0] > 0.0:
                if target[2] > 0.04:
                    self.T_target[:3, 3] = target
                    appropriate_target = True
        print("Target :", self.T_target[:3, 3])
        return

    def move_to_target(self, rollout_num):
        """Move robot to target.

        [Resolved Rate Motion Control]
          - material: https://www.youtube.com/embed/rkHs7K0ad14?rel=0&showinfo=0

        q_now: Current joint angles.
        T_cur: Current transformation matrix.

        1) Get current time
        2) Get gripper position and orientation by calculate forward kinematics
           of current joint angles.
        3) Path planning by cubic function. Set goal matrix.
        4) Get jacobian .
        5) Get q_new by inverse term.
        6) Scaling joint velocities.
        7) Set joint states.
        8) Save file.
        """
        t_now = rospy.get_rostime().secs + rospy.get_rostime().nsecs * 10 ** -9
        # TODO: why mutex needed?
        with self.mutex:
            q_now = self.q

        self.T_cur = np.array(self.robot.forward(q_now))
        self._gripper_pos = self.T_cur[:3, 3]
        self._gripper_orientation[3] = (
            1 + self.T_cur[0, 0] + self.T_cur[1, 1] + self.T_cur[2, 2]
        ) ** 0.5
        self._gripper_orientation[0] = (self.T_cur[2, 1] - self.T_cur[1, 2]) / (
            4 * self._gripper_orientation[3]
        )
        self._gripper_orientation[1] = (self.T_cur[0, 2] - self.T_cur[2, 0]) / (
            4 * self._gripper_orientation[3]
        )
        self._gripper_orientation[2] = (self.T_cur[1, 0] - self.T_cur[0, 1]) / (
            4 * self._gripper_orientation[3]
        )

        # implement multi-array cubic calculation
        for i in range(3):
            self.T_goal[i, 3] = self.cubic(
                t_now,
                self.control_start_time,
                self.control_start_time + 2.0,
                self.T_init[i, 3],
                self.T_target[i, 3],
                0.0,
                0.0,
            )

        e = self.T_goal[:3, 3] - self.T_cur[:3, 3]

        Jb = np.array(self.robot.jacobian(q_now))
        Jv = Jb[:3, :]

        invterm = np.linalg.inv(np.dot(Jv, Jv.T) + pow(self.damping, 2) * np.eye(3))
        kp = 2.0
        qdot_new = np.dot(np.dot(Jv.T, invterm), kp * e)

        # Scaling joint velocity
        def _limit_q_dot(qdot_new):
            minus_v = abs(np.amin(qdot_new))
            plus_v = abs(np.amax(qdot_new))
            if minus_v > plus_v:
                scale = minus_v
            else:
                scale = plus_v
            if scale > self.joint_vel_limit:
                qdot_new = qdot_new / scale * self.joint_vel_limit
            return qdot_new

        self.qdot = _limit_q_dot(qdot_new)

        dt = 1.0 / self.hz
        self.q_desired = self.q_desired + qdot_new * dt
        self.q_desired = self.joint_limit_check(self.q_desired)

        self.j1_pos_command_pub.publish(self.q_desired[0])
        self.j2_pos_command_pub.publish(self.q_desired[1])
        self.j3_pos_command_pub.publish(self.q_desired[2])
        self.j4_pos_command_pub.publish(self.q_desired[3])
        self.joint_pos_command_to_dxl_pub.publish(data=self.q_desired)

        obs = np.concatenate(
            (
                self._gripper_pos,
                self._gripper_orientation,
                self.q,
                self.qdot,
                self.effort,
            )
        )

        # append data
        self.data[rollout_num]["observation"].append(obs.tolist())
        self.data[rollout_num]["desired q"].append(self.q_desired.tolist())
        self.data[rollout_num]["target"].append(self.T_target[:3, 3].tolist())

        def _is_done_move_to_target():
            if np.mean(np.abs(self.T_target[:3, 3] - self.T_cur[:3, 3])) < 0.001:
                self.q_init = list(self.q)
                print("Target arrived!")
                return True
            else:
                return False

        self.done_move_to_target = _is_done_move_to_target()

    def move_to_init(self):
        """Move robot to initial pose."""
        t_now = self.get_rostime()
        for i in range(4):
            self.q_desired[i] = self.cubic(
                t_now,
                self.control_start_time,
                self.control_start_time + 3.0,
                self.q_init[i],
                0.0,
                0.0,
                0.0,
            )
        self.q_desired = self.joint_limit_check(self.q_desired)

        self.j1_pos_command_pub.publish(self.q_desired[0])
        self.j2_pos_command_pub.publish(self.q_desired[1])
        self.j3_pos_command_pub.publish(self.q_desired[2])
        self.j4_pos_command_pub.publish(self.q_desired[3])
        self.joint_pos_command_to_dxl_pub.publish(data=self.q_desired)

        def _is_done_init():
            if np.mean(np.abs(np.zeros(4) - self.q)) < 0.05:
                time.sleep(2.0)
                print("Initial Pose Arrived!")
                return True
            else:
                return False

        self.done_init = _is_done_init()

    def joint_limit_check(self, q_target):
        q_limit_L = [-pi * 0.9, -pi * 0.57, -pi * 0.3, -pi * 0.57]
        q_limit_H = [pi * 0.9, pi * 0.5, pi * 0.44, pi * 0.65]
        for i in range(4):
            if q_target[i] < q_limit_L[i]:
                q_target[i] = q_limit_L[i]
            elif q_target[i] > q_limit_H[i]:
                q_target[i] = q_limit_H[i]
        return q_target

    def cubic(self, t, t_0, t_f, x_0, x_f, x_dot_0, x_dot_f):
        """
        [Cubic polynomials]
          - material: http://ocw.snu.ac.kr/sites/default/files/NOTE/Chap07_Trajectory%20generation.pdf

        t: time
        t_0: init time
        t_f: final time
        x_0: init position
        x_f: final position
        x_dot_0: velocity when x_0
        x_dot_f: velocity when x_f
        """
        if t < t_0:
            x_t = x_0  # theta(0)
        elif t > t_f:
            x_t = x_f  # theta(t_f)
        else:
            total_x = x_f - x_0
            elapsed_t = t - t_0  # t
            total_t = t_f - t_0  # t_f

            x_t = (x_0 + x_dot_0 * elapsed_t
                   + (3 * total_x / total_t**2 - 2 * x_dot_0 / total_t - x_dot_f / total_t) * elapsed_t**2
                   + (-2 * total_x / total_t**3 + (x_dot_0 + x_dot_f) / total_t**2) * elapsed_t**3)
        return x_t


def main():
    rospy.init_node("demo_collector")
    try:
        demo_collector = DemoCollector()
        demo_collector.run()
    except rospy.ROSInterruptException:
        pass

    rospy.spin()


if __name__ == "__main__":
    main()
