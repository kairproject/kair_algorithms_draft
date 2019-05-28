from math import pi


def deg2rad(deg):
    return deg * pi / 180


def rad2deg(rad):
    return rad * 180 / pi


def rpm2rad(rpm):
    return 2 * rpm * pi / 60
