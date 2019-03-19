#!/usr/bin/env python

from envs.open_manipulator import OpenManipulatorEnv  # noqa


def test_reset():
    # assert obs in specific boundary
    pass


def test_forward():
    # define actions
    # assert obs in specific boundary
    pass


def test_rotate():
    # define actions
    # assert obs in specific boundary
    pass


def test_block_loc():
    # block generation code
    # assert block in specific boundary (gripper's movable area)
    pass


def test_achieve_goal():
    # define actions
    # define goal
    # assert gripper reach goal
    pass


if __name__ == '__main__':
    test_reset()
    test_forward()
    test_rotate()
    test_block_loc()
    test_achieve_goal()
