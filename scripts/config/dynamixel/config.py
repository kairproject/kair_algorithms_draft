config = {
    "DXL_RESOLUTION": 0.088,  # Position resolution of XM430-W210 in degree
    "DXL_VELOCITY_RESOLUTION": 0.229,  # Velocity resolition of XM430-W210 in rpm
    "DXL_TO_CURRENT": 2.69,  # From dynamixel return value to current
    # Control table address
    "ADDR_TORQUE_ENABLE": 64,  # To set torque on/off
    "ADDR_PRESENT_POSITION": 132,  # To read position
    "ADDR_PRESENT_VELOCITY": 128,  # To read velocity
    "ADDR_PRESENT_CURRENT": 126,  # To read current
    "ADDR_OP_MODE": 11,  # To set operation mode (position/velocity/multi_turn mode)
    "ADDR_GOAL_POSITION": 116,  # To write position
    # Data Byte Length
    "LEN_GOAL_POSITION": 4,  # In byte
    "LEN_PRESENT_POSITION": 4,  # In byte
    "LEN_PRESENT_VELOCITY": 4,  # In byte
    "LEN_PRESENT_CURRENT": 2,  # In byte
    "CW_LIMIT": 4095,  # Clockwise limit
    "CCW_LIMIT": 0,  # Counter clock wise limit
    "DXL_POS_OFFSET": 2048,  # Initial position
    # Protocol version
    "PROTOCOL_VERSION": 2.0,
    # Default setting
    "DXL1_ID": 11,  # Joint 1 ID
    "DXL2_ID": 12,  # Joint 2 ID
    "DXL3_ID": 13,  # Joint 3 ID
    "DXL4_ID": 14,  # Joint 4 ID
    "BAUDRATE": 1000000,
    "DEVICENAME": "/dev/ttyUSB0",  # Connected USB port
    "TORQUE_ENABLE": 1,  # Torque on
    "TORQUE_DISABLE": 0,  # Torque off
}
