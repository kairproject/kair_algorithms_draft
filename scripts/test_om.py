#!/usr/bin/env python

from math import pi, sin

import numpy as np

import rospy
from envs.open_manipulator import OpenManipulatorEnv


if __name__ == "__main__":
    env = OpenManipulatorEnv()
    _ = env.reset()
    r = rospy.Rate(100)
    for i in range(2000):
        act = -2 * pi * sin(i / 2000.0) * np.ones(6)
        _, _, _ = env.step(action=act, step=i)
    r.sleep()
