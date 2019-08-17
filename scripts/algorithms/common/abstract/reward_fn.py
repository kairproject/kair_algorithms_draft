# -*- coding: utf-8 -*-
"""Abstract class for computing reward.
- Author: Kh Kim
- Contact: kh.kim@medipixel.io
"""

from abc import ABCMeta, abstractmethod


class RewardFn(object):
    """Abstract class for computing reward.
       New compute_reward class should redefine __call__()
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def __call__(self, transition, goal_state):
        pass
