#!/usr/bin/env python

import numpy as np

# ROS Imports
import rospy
from config import config as cfg
from dynamixel_sdk import (
    DXL_HIBYTE,
    DXL_HIWORD,
    DXL_LOBYTE,
    DXL_LOWORD,
    GroupBulkRead,
    GroupSyncWrite,
    PacketHandler,
    PortHandler,
)
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from utils import deg2rad, rad2deg, rpm2rad


class DynamixelPositionControl(object):
    def __init__(self, cfg):
        # Dynamixel Setting
        rospy.loginfo("Dynamixel Position Controller Created")
        self.cfg = cfg
        self.portHandler = PortHandler(self.cfg["DEVICENAME"])
        self.packetHandler = PacketHandler(self.cfg["PROTOCOL_VERSION"])
        self.groupSyncWrite = GroupSyncWrite(
            self.portHandler,
            self.packetHandler,
            self.cfg["ADDR_GOAL_POSITION"],
            self.cfg["LEN_GOAL_POSITION"],
        )
        self.groupBulkReadPosition = GroupBulkRead(self.portHandler, self.packetHandler)
        self.groupBulkReadVelocity = GroupBulkRead(self.portHandler, self.packetHandler)
        self.groupBulkReadCurrent = GroupBulkRead(self.portHandler, self.packetHandler)
        # Port Open
        if self.portHandler.openPort():
            print("Succeeded to open the port")
        else:
            print("Failed to open the port")
            quit()

        # Set port baudrate
        if self.portHandler.setBaudRate(self.cfg["BAUDRATE"]):
            print("Succeeded to change the baudrate")
        else:
            print("Failed to change the baudrate")
            quit()

        self.packetHandler.write1ByteTxRx(
            self.portHandler, self.cfg["DXL1_ID"], self.cfg["ADDR_OP_MODE"], 3
        )
        self.packetHandler.write1ByteTxRx(
            self.portHandler, self.cfg["DXL2_ID"], self.cfg["ADDR_OP_MODE"], 3
        )
        self.packetHandler.write1ByteTxRx(
            self.portHandler, self.cfg["DXL3_ID"], self.cfg["ADDR_OP_MODE"], 3
        )
        self.packetHandler.write1ByteTxRx(
            self.portHandler, self.cfg["DXL4_ID"], self.cfg["ADDR_OP_MODE"], 3
        )

        self.groupBulkReadPosition.addParam(
            self.cfg["DXL1_ID"],
            self.cfg["ADDR_PRESENT_POSITION"],
            self.cfg["LEN_PRESENT_POSITION"],
        )
        self.groupBulkReadPosition.addParam(
            self.cfg["DXL2_ID"],
            self.cfg["ADDR_PRESENT_POSITION"],
            self.cfg["LEN_PRESENT_POSITION"],
        )
        self.groupBulkReadPosition.addParam(
            self.cfg["DXL3_ID"],
            self.cfg["ADDR_PRESENT_POSITION"],
            self.cfg["LEN_PRESENT_POSITION"],
        )
        self.groupBulkReadPosition.addParam(
            self.cfg["DXL4_ID"],
            self.cfg["ADDR_PRESENT_POSITION"],
            self.cfg["LEN_PRESENT_POSITION"],
        )
        self.groupBulkReadVelocity.addParam(
            self.cfg["DXL1_ID"],
            self.cfg["ADDR_PRESENT_VELOCITY"],
            self.cfg["LEN_PRESENT_VELOCITY"],
        )
        self.groupBulkReadVelocity.addParam(
            self.cfg["DXL2_ID"],
            self.cfg["ADDR_PRESENT_VELOCITY"],
            self.cfg["LEN_PRESENT_VELOCITY"],
        )
        self.groupBulkReadVelocity.addParam(
            self.cfg["DXL3_ID"],
            self.cfg["ADDR_PRESENT_VELOCITY"],
            self.cfg["LEN_PRESENT_VELOCITY"],
        )
        self.groupBulkReadVelocity.addParam(
            self.cfg["DXL4_ID"],
            self.cfg["ADDR_PRESENT_VELOCITY"],
            self.cfg["LEN_PRESENT_VELOCITY"],
        )

        self.groupBulkReadCurrent.addParam(
            self.cfg["DXL1_ID"],
            self.cfg["ADDR_PRESENT_CURRENT"],
            self.cfg["LEN_PRESENT_CURRENT"],
        )
        self.groupBulkReadCurrent.addParam(
            self.cfg["DXL2_ID"],
            self.cfg["ADDR_PRESENT_CURRENT"],
            self.cfg["LEN_PRESENT_CURRENT"],
        )
        self.groupBulkReadCurrent.addParam(
            self.cfg["DXL3_ID"],
            self.cfg["ADDR_PRESENT_CURRENT"],
            self.cfg["LEN_PRESENT_CURRENT"],
        )
        self.groupBulkReadCurrent.addParam(
            self.cfg["DXL4_ID"],
            self.cfg["ADDR_PRESENT_CURRENT"],
            self.cfg["LEN_PRESENT_CURRENT"],
        )

        # Enable Dynamixel Torque
        self.packetHandler.write1ByteTxRx(
            self.portHandler,
            self.cfg["DXL1_ID"],
            self.cfg["ADDR_TORQUE_ENABLE"],
            self.cfg["TORQUE_ENABLE"],
        )
        self.packetHandler.write1ByteTxRx(
            self.portHandler,
            self.cfg["DXL2_ID"],
            self.cfg["ADDR_TORQUE_ENABLE"],
            self.cfg["TORQUE_ENABLE"],
        )
        self.packetHandler.write1ByteTxRx(
            self.portHandler,
            self.cfg["DXL3_ID"],
            self.cfg["ADDR_TORQUE_ENABLE"],
            self.cfg["TORQUE_ENABLE"],
        )
        self.packetHandler.write1ByteTxRx(
            self.portHandler,
            self.cfg["DXL4_ID"],
            self.cfg["ADDR_TORQUE_ENABLE"],
            self.cfg["TORQUE_ENABLE"],
        )

        # ROS Publisher
        self.joint_states_pub = rospy.Publisher(
            "/open_manipulator/joint_states_real", JointState, queue_size=3
        )
        # ROS Subcriber
        self.joint_pos_command_sub = rospy.Subscriber(
            "/open_manipulator/joint_position/command",
            Float64MultiArray,
            self.joint_command_cb,
        )

        self.joint_states = JointState()
        self.dxl_present_position = np.zeros(4)
        self.dxl_present_velocity = np.zeros(4)
        self.dxl_present_current = np.zeros(4)
        self.q_desired = np.zeros(4)
        self.dxl_goal_position = [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        self.read_dxl()
        for i in range(4):
            self.dxl_goal_position[i] = [
                DXL_LOBYTE(DXL_LOWORD(int(self.dxl_present_position[i]))),
                DXL_HIBYTE(DXL_LOWORD(int(self.dxl_present_position[i]))),
                DXL_LOBYTE(DXL_HIWORD(int(self.dxl_present_position[i]))),
                DXL_HIBYTE(DXL_HIWORD(int(self.dxl_present_position[i]))),
            ]

        self.r = rospy.Rate(100)
        try:
            while not rospy.is_shutdown():
                self.read_dxl()
                self.write_dxl()
                self.r.sleep()
        except KeyboardInterrupt:
            self.packetHandler.write1ByteTxRx(
                self.portHandler,
                self.cfg["DXL1_ID"],
                self.cfg["ADDR_TORQUE_ENABLE"],
                self.cfg["TORQUE_DISABLE"],
            )
            self.packetHandler.write1ByteTxRx(
                self.portHandler,
                self.cfg["DXL2_ID"],
                self.cfg["ADDR_TORQUE_ENABLE"],
                self.cfg["TORQUE_DISABLE"],
            )
            self.packetHandler.write1ByteTxRx(
                self.portHandler,
                self.cfg["DXL3_ID"],
                self.cfg["ADDR_TORQUE_ENABLE"],
                self.cfg["TORQUE_DISABLE"],
            )
            self.packetHandler.write1ByteTxRx(
                self.portHandler,
                self.cfg["DXL4_ID"],
                self.cfg["ADDR_TORQUE_ENABLE"],
                self.cfg["TORQUE_DISABLE"],
            )

    def joint_command_cb(self, joint_desired):
        i = 0
        while i < 4:
            self.q_desired[i] = joint_desired.data[i]
            dxl_command = int(
                rad2deg(self.q_desired[i]) / self.cfg["DXL_RESOLUTION"]
                + self.cfg["DXL_POS_OFFSET"]
            )
            if dxl_command > self.cfg["CW_LIMIT"]:
                dxl_command = self.cfg["CW_LIMIT"]
            elif dxl_command < self.cfg["CCW_LIMIT"]:
                dxl_command = self.cfg["CCW_LIMIT"]

            self.dxl_goal_position[i] = [
                DXL_LOBYTE(DXL_LOWORD(dxl_command)),
                DXL_HIBYTE(DXL_LOWORD(dxl_command)),
                DXL_LOBYTE(DXL_HIWORD(dxl_command)),
                DXL_HIBYTE(DXL_HIWORD(dxl_command)),
            ]
            i += 1

    def read_dxl(self):
        self.groupBulkReadPosition.txRxPacket()

        self.dxl_present_position[0] = self.groupBulkReadPosition.getData(
            self.cfg["DXL1_ID"],
            self.cfg["ADDR_PRESENT_POSITION"],
            self.cfg["LEN_PRESENT_POSITION"],
        )
        self.dxl_present_position[1] = self.groupBulkReadPosition.getData(
            self.cfg["DXL2_ID"],
            self.cfg["ADDR_PRESENT_POSITION"],
            self.cfg["LEN_PRESENT_POSITION"],
        )
        self.dxl_present_position[2] = self.groupBulkReadPosition.getData(
            self.cfg["DXL3_ID"],
            self.cfg["ADDR_PRESENT_POSITION"],
            self.cfg["LEN_PRESENT_POSITION"],
        )
        self.dxl_present_position[3] = self.groupBulkReadPosition.getData(
            self.cfg["DXL4_ID"],
            self.cfg["ADDR_PRESENT_POSITION"],
            self.cfg["LEN_PRESENT_POSITION"],
        )

        self.groupBulkReadVelocity.txRxPacket()
        self.dxl_present_velocity[0] = self.groupBulkReadVelocity.getData(
            self.cfg["DXL1_ID"],
            self.cfg["ADDR_PRESENT_VELOCITY"],
            self.cfg["LEN_PRESENT_VELOCITY"],
        )
        self.dxl_present_velocity[1] = self.groupBulkReadVelocity.getData(
            self.cfg["DXL2_ID"],
            self.cfg["ADDR_PRESENT_VELOCITY"],
            self.cfg["LEN_PRESENT_VELOCITY"],
        )
        self.dxl_present_velocity[2] = self.groupBulkReadVelocity.getData(
            self.cfg["DXL3_ID"],
            self.cfg["ADDR_PRESENT_VELOCITY"],
            self.cfg["LEN_PRESENT_VELOCITY"],
        )
        self.dxl_present_velocity[3] = self.groupBulkReadVelocity.getData(
            self.cfg["DXL4_ID"],
            self.cfg["ADDR_PRESENT_VELOCITY"],
            self.cfg["LEN_PRESENT_VELOCITY"],
        )

        self.groupBulkReadCurrent.txRxPacket()
        self.dxl_present_current[0] = self.groupBulkReadVelocity.getData(
            self.cfg["DXL1_ID"],
            self.cfg["ADDR_PRESENT_CURRENT"],
            self.cfg["LEN_PRESENT_CURRENT"],
        )
        self.dxl_present_current[1] = self.groupBulkReadVelocity.getData(
            self.cfg["DXL2_ID"],
            self.cfg["ADDR_PRESENT_CURRENT"],
            self.cfg["LEN_PRESENT_CURRENT"],
        )
        self.dxl_present_current[2] = self.groupBulkReadVelocity.getData(
            self.cfg["DXL3_ID"],
            self.cfg["ADDR_PRESENT_CURRENT"],
            self.cfg["LEN_PRESENT_CURRENT"],
        )
        self.dxl_present_current[3] = self.groupBulkReadVelocity.getData(
            self.cfg["DXL4_ID"],
            self.cfg["ADDR_PRESENT_CURRENT"],
            self.cfg["LEN_PRESENT_CURRENT"],
        )

        for i in range(4):
            if self.dxl_present_velocity[i] > 2 ** (
                8 * self.cfg["ADDR_PRESENT_VELOCITY"] / 2
            ):
                self.dxl_present_velocity[i] = self.dxl_present_velocity[i] - 2 ** (
                    8 * self.cfg["ADDR_PRESENT_VELOCITY"]
                )
            if self.dxl_present_current[i] > 2 ** (
                8 * self.cfg["LEN_PRESENT_CURRENT"] / 2
            ):
                self.dxl_present_current[i] = self.dxl_present_current[i] - 2 ** (
                    8 * self.cfg["LEN_PRESENT_CURRENT"]
                )

        q_current = [
            0.0,
            0.0,
            deg2rad(
                (self.dxl_present_position[0] - self.cfg["DXL_POS_OFFSET"])
                * self.cfg["DXL_RESOLUTION"]
            ),
            deg2rad(
                (self.dxl_present_position[1] - self.cfg["DXL_POS_OFFSET"])
                * self.cfg["DXL_RESOLUTION"]
            ),
            deg2rad(
                (self.dxl_present_position[2] - self.cfg["DXL_POS_OFFSET"])
                * self.cfg["DXL_RESOLUTION"]
            ),
            deg2rad(
                (self.dxl_present_position[3] - self.cfg["DXL_POS_OFFSET"])
                * self.cfg["DXL_RESOLUTION"]
            ),
        ]
        qdot_current = [
            0.0,
            0.0,
            rpm2rad(self.dxl_present_velocity[0] * self.cfg["DXL_VELOCITY_RESOLUTION"]),
            rpm2rad(self.dxl_present_velocity[1] * self.cfg["DXL_VELOCITY_RESOLUTION"]),
            rpm2rad(self.dxl_present_velocity[2] * self.cfg["DXL_VELOCITY_RESOLUTION"]),
            rpm2rad(self.dxl_present_velocity[3] * self.cfg["DXL_VELOCITY_RESOLUTION"]),
        ]
        motor_current = [
            0.0,
            0.0,
            self.dxl_present_current[0] * self.cfg["DXL_TO_CURRENT"],
            self.dxl_present_current[1] * self.cfg["DXL_TO_CURRENT"],
            self.dxl_present_current[2] * self.cfg["DXL_TO_CURRENT"],
            self.dxl_present_current[3] * self.cfg["DXL_TO_CURRENT"],
        ]

        self.joint_states.position = q_current
        self.joint_states.velocity = qdot_current
        self.joint_states.effort = motor_current

        self.joint_states_pub.publish(self.joint_states)

    def write_dxl(self):
        self.groupSyncWrite.addParam(self.cfg["DXL1_ID"], self.dxl_goal_position[0])
        self.groupSyncWrite.addParam(self.cfg["DXL2_ID"], self.dxl_goal_position[1])
        self.groupSyncWrite.addParam(self.cfg["DXL3_ID"], self.dxl_goal_position[2])
        self.groupSyncWrite.addParam(self.cfg["DXL4_ID"], self.dxl_goal_position[3])

        self.groupSyncWrite.txPacket()
        self.groupSyncWrite.clearParam()

    def error_check(self, dxl_comm_result, dxl_error):
        if dxl_comm_result != self.cfg["COMM_SUCCESS"]:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))


def main():
    rospy.init_node("DXL_pos_control")

    try:
        DynamixelPositionControl(cfg)
    except rospy.ROSInterruptException:
        pass

    rospy.spin()


if __name__ == "__main__":
    main()
