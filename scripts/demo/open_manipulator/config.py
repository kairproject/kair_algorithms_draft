config = {
    # pos_read_write
    # TODO : Change Control Table to XM430-W350 spec ###
    "DXL_RESOLUTION": 0.088,  # In degree
    "DXL_VELOCITY_RESOLUTION": 0.229,  # In rpm # For XM430-W210 0.229
    "DXL_TO_CURRENT": 2.69,  # 2.69 mA
    # Control table address
    "ADDR_TORQUE_ENABLE": 64,
    "ADDR_PRESENT_POSITION": 132,
    "ADDR_PRESENT_VELOCITY": 128,
    "ADDR_PRESENT_CURRENT": 126,
    "ADDR_OP_MODE": 11,
    "ADDR_GOAL_POSITION": 116,
    # Data Byte Length
    "LEN_GOAL_POSITION": 4,
    "LEN_PRESENT_POSITION": 4,
    "LEN_PRESENT_VELOCITY": 4,
    "LEN_PRESENT_CURRENT": 2,
    "CW_LIMIT": 4095,
    "CCW_LIMIT": 0,
    "DXL_POS_OFFSET": 2048,
    # Protocol version
    "PROTOCOL_VERSION": 2.0,
    # Default setting
    "DXL1_ID": 11,
    "DXL2_ID": 12,
    "DXL3_ID": 13,
    "DXL4_ID": 14,
    "BAUDRATE": 1000000,
    "DEVICENAME": "/dev/ttyUSB0",
    "TORQUE_ENABLE": 1,
    "TORQUE_DISABLE": 0,
    # Collector
    "DAMPING": 0.02,
    "JOINT_VEL_LIMIT": 4,
}
