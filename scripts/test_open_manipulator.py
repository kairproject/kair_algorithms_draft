#!/usr/bin/env python

from envs.open_manipulator import OpenManipulatorEnv


def test_reset():
    env = OpenManipulatorEnv()
    obs = env.reset()
    # assert obs in specific boundary


def test_forward():
    env = OpenManipulatorEnv()
    obs = env.reset()
    # define actions
    # assert obs in specific boundary


def test_rotate():
    env = OpenManipulatorEnv()
    obs = env.reset()
    # define actions
    # assert obs in specific boundary


def test_block_loc():
    env = OpenManipulatorEnv()
    obs = env.reset()
    # block generation code
    # assert block in specific boundary (gripper's movable area)


def test_achieve_goal():
    env = OpenManipulatorEnv()
    obs = env.reset()
    # define actions
    # define goal
    # assert gripper reach goal


if __name__ == '__main__':
    test_reset()
    test_forward()
    test_rotate()
    test_block_loc()
    test_achieve_goal()
    obs = test_reset()
    print("test_reset:", obs)
