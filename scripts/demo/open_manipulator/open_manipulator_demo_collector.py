#!/usr/bin/env python
import json
import random
import threading
import time
from collections import OrderedDict
from math import pi, pow

import numpy as np


# ROS Imports
import rospy
from pykdl_utils.kdl_kinematics import KDLKinematics
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Float64MultiArray
from urdf_parser_py.urdf import URDF

####################
# GLOBAL VARIABLES #
####################
DAMPING = 0.01  # 0.00
JOINT_VEL_LIMIT = 4  # 2rad/s


class DemoCollector(object):
    def __init__(self):

        rospy.loginfo("Start Demo Collector")
        self.use_platform = rospy.get_param("~use_platform")
        self.robot_urdf = URDF.from_parameter_server()
        self.robot = KDLKinematics(self.robot_urdf, "world", "end_effector_link")

        # Shared variables
        self.mutex = threading.Lock()
        self.damping = rospy.get_param("~damping", DAMPING)
        self.joint_vel_limit = rospy.get_param("~joint_vel_limit", JOINT_VEL_LIMIT)
        self.q = np.zeros(4)  # Joint angles
        self.q_desired = np.zeros(4)
        self.qdot = np.zeros(4)  # Joint velocities
        self.effort = np.zeros(4)  # Joint torque
        self.T_target = np.array(self.robot.forward(self.q))
        self.T_goal = np.array(self.robot.forward(self.q))
        self.T_cur = np.array(self.robot.forward(self.q))

        # Observation
        self._gripper_pos = np.zeros(3)
        self._gripper_orientation = np.zeros(4)

        self.init = False
        self.is_joint_states_cb = False
        self.is_set_new_target = False
        self.is_init_pos = False
        self.is_finished = False

        self.num_tar_demo = 10
        self.num_cur_demo = 0

        self.control_start_time = (
            rospy.get_rostime().secs + rospy.get_rostime().nsecs * 10 ** -9
        )

        # Subscriber
        if self.use_platform is False:
            self.joint_states_sub = rospy.Subscriber(
                "/open_manipulator/joint_states", JointState, self.joint_states_cb
            )
        else:
            self.joint_states_sub = rospy.Subscriber(
                "/open_manipulator/joint_states_real", JointState, self.joint_states_cb
            )

        # Command publisher
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

        self.r = rospy.Rate(100)
        while not self.is_finished:
            if self.is_joint_states_cb is True:
                if self.init is False:
                    rospy.loginfo("Moving to Initial Position")
                    self.q_init = list(self.q)
                    self.control_start_time = (
                        rospy.get_rostime().secs + rospy.get_rostime().nsecs * 10 ** -9
                    )
                    self.init = True

                if self.is_set_new_target is False and self.is_init_pos:
                    self.start_log()
                    self.set_target()
                    self.T_init = np.array(self.robot.forward(self.q))
                    self.control_start_time = (
                        rospy.get_rostime().secs + rospy.get_rostime().nsecs * 10 ** -9
                    )
                    self.is_set_new_target = True
                    self.num_cur_demo = self.num_cur_demo + 1
                if self.is_set_new_target is True:
                    self.move_to_target()
                if self.is_init_pos is False:
                    self.move_to_init()

            if self.is_set_new_target is True or self.is_init_pos is False:
                self.r.sleep()
            if self.num_cur_demo > self.num_tar_demo:
                print("Demo Collection Finished!")
                self.is_finished = True

    def joint_states_cb(self, joint_states):
        self.is_joint_states_cb = True
        i = 0
        while i < 4:
            self.q[i] = joint_states.position[i + 2]
            self.qdot[i] = joint_states.velocity[i + 2]
            self.effort[i] = joint_states.effort[i + 2]
            i += 1

    def start_log(self):
        self.f = open("../DemoEpisode" + str(self.num_cur_demo) + ".txt", "w")

    def set_target(self):
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
            target = np.empty_like(self.T_target[0:3, 3])
            target[:] = self.T_target[0:3, 3]

            min_op_distance = 0.15
            max_op_distance = 0.4

            if np.linalg.norm(np.abs(target)) > max_op_distance:
                target = target * max_op_distance / np.linalg.norm(np.abs(target))

            if np.linalg.norm(np.abs(target)) < min_op_distance:
                target = target * min_op_distance / np.linalg.norm(np.abs(target))

            if target[0] > 0.0:
                if target[2] > 0.04:
                    self.T_target[0:3, 3] = target
                    appropriate_target = True
        print("Episode ", self.num_cur_demo)
        print("Target :", self.T_target[0:3, 3])
        return

    def move_to_target(self):
        t_now = rospy.get_rostime().secs + rospy.get_rostime().nsecs * 10 ** -9
        with self.mutex:
            q_now = self.q

        self.T_cur = np.array(self.robot.forward(q_now))
        self._gripper_pos = self.T_cur[0:3, 3]
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

        e = self.T_goal[0:3, 3] - self.T_cur[0:3, 3]

        Jb = np.array(self.robot.jacobian(q_now))
        Jv = Jb[0:3, :]

        invterm = np.linalg.inv(np.dot(Jv, Jv.T) + pow(self.damping, 2) * np.eye(3))
        kp = 2.0
        qdot_new = np.dot(np.dot(Jv.T, invterm), kp * e)

        # Scaling joint velocity
        minus_v = abs(np.amin(qdot_new))
        plus_v = abs(np.amax(qdot_new))
        if minus_v > plus_v:
            scale = minus_v
        else:
            scale = plus_v
        if scale > self.joint_vel_limit:
            qdot_new = 2.0 * (qdot_new / scale) * self.joint_vel_limit
        self.qdot = qdot_new

        dt = 0.01
        self.q_desired = self.q_desired + qdot_new * dt
        self.q_desired = self.joint_limit_check(self.q_desired)

        self.j1_pos_command_pub.publish(self.q_desired[0])
        self.j2_pos_command_pub.publish(self.q_desired[1])
        self.j3_pos_command_pub.publish(self.q_desired[2])
        self.j4_pos_command_pub.publish(self.q_desired[3])
        self.joint_pos_command_to_dxl_pub.publish(data=self.q_desired)

        data = OrderedDict()
        obs = np.concatenate(
            (
                self._gripper_pos,
                self._gripper_orientation,
                self.q,
                self.qdot,
                self.effort,
            )
        )
        data["observation"] = obs.tolist()
        data["desired q"] = self.q_desired.tolist()
        data["target"] = self.T_target[0:3, 3].tolist()
        json.dump(data, self.f, ensure_ascii=False)

        if np.mean(np.abs(self.T_target[0:3, 3] - self.T_cur[0:3, 3])) < 0.001:
            self.is_set_new_target = False
            self.is_init_pos = False
            self.q_init = list(self.q)
            self.control_start_time = (
                rospy.get_rostime().secs + rospy.get_rostime().nsecs * 10 ** -9
            )
            self.f.close()
            print("Target arrived!")

        return

    def move_to_init(self):
        t_now = rospy.get_rostime().secs + rospy.get_rostime().nsecs * 10 ** -9
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

        if np.mean(np.abs(np.zeros(4) - self.q)) < 0.05:
            time.sleep(2.0)
            self.is_init_pos = True
            print("Initial Pose Arrived!")

    def joint_limit_check(self, q_target):
        q_limit_L = [-pi * 0.9, -pi * 0.57, -pi * 0.3, -pi * 0.57]
        q_limit_H = [pi * 0.9, pi * 0.5, pi * 0.44, pi * 0.65]
        for i in range(4):
            if q_target[i] < q_limit_L[i]:
                q_target[i] = q_limit_L[i]
            elif q_target[i] > q_limit_H[i]:
                q_target[i] = q_limit_H[i]
        return q_target

    def cubic(self, time, time_0, time_f, x_0, x_f, x_dot_0, x_dot_f):
        x_t = x_0

        if time < time_0:
            x_t = x_0

        elif time > time_f:
            x_t = x_f
        else:
            elapsed_time = time - time_0
            total_time = time_f - time_0
            total_time2 = total_time * total_time
            total_time3 = total_time2 * total_time
            total_x = x_f - x_0

            x_t = (
                x_0
                + x_dot_0 * elapsed_time
                + (
                    3 * total_x / total_time2
                    - 2 * x_dot_0 / total_time
                    - x_dot_f / total_time
                )
                * elapsed_time
                * elapsed_time
                + (-2 * total_x / total_time3 + (x_dot_0 + x_dot_f) / total_time2)
                * elapsed_time
                * elapsed_time
                * elapsed_time
            )

        return x_t


def main():
    rospy.init_node("demo_collector")
    try:
        DemoCollector()
    except rospy.ROSInterruptException:
        pass

    rospy.spin()


if __name__ == "__main__":
    main()
