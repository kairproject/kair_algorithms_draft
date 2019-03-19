#!/usr/bin/env python

import numpy as np
import time as timer
import time
from math import pi, cos, sin, radians
from open_manipulator import OpenManipulatorEnv
import rospy


import pickle
import os


if __name__ == '__main__':
    env = OpenManipulatorEnv()
    _ = env.reset()
    r = rospy.Rate(100)
    for i in range(2000):

        act = -2*pi*sin(i/2000.0)*np.ones(6)

        _, _, _ = env.step(action = act, step=i)
    r.sleep()
