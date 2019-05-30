"""Utils for dynamixel.

- Author: DH Kim
- Contact: kdh0429@snu.ac.kr
"""
from math import pi


def deg2rad(deg):
    """ Transform degree to radian."""
    return deg * pi / 180


def rad2deg(rad):
    """ Transform radian to degree."""
    return rad * 180 / pi


def rpm2rad(rpm):
    """ Transform RPM to radian."""
    return 2 * rpm * pi / 60
