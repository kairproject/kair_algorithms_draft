# -*- coding: utf-8 -*-
"""Abstract class used for Hindsight Experience Replay.
- Author: Kh Kim
- Contact: kh.kim@medipixel.io
- Paper: https://arxiv.org/pdf/1707.01495.pdf
"""

from abc import ABCMeta, abstractmethod

import numpy as np


class HER(object):
    """Abstract class for HER (final strategy).
    Attributes:
        reward_func (Callable): returns reward from state, action, next_state
    """

    __metaclass__ = ABCMeta

    def __init__(self, reward_func):
        """Initialization.

        Args:
            reward_func (Callable): returns reward from state, action, next_state
        """
        self.reward_func = reward_func()

    @abstractmethod
    def fetch_desired_states_from_demo(self, demo):
        pass

    @abstractmethod
    def get_desired_state(self, *args):
        pass

    @abstractmethod
    def generate_demo_transitions(self, demo):
        pass

    @abstractmethod
    def _get_final_state(self, transition):
        pass

    def _append_origin_transitions(self, origin_transitions, transition, desired_state):
        """Append original transitions adding goal state for training."""
        origin_transitions.append(self._get_transition(transition, desired_state))

    def _append_new_transitions(self, new_transitions, transition, final_state):
        """Append new transitions made by HER strategy (final) for training."""
        new_transitions.append(self._get_transition(transition, final_state))

    def _get_transition(self, transition, goal_state):
        """Get a single transition concatenated with a goal state."""
        state, action, _, next_state, done = transition

        done = np.array_equal(next_state, goal_state)
        reward = self.reward_func(transition, goal_state)
        state = np.concatenate((state, goal_state), axis=-1)
        next_state = np.concatenate((next_state, goal_state), axis=-1)

        return state, action, reward, next_state, done

    def generate_transitions(
        self, transitions, desired_state, success_score, is_demo=False
    ):
        """Generate new transitions concatenated with desired states."""
        origin_transitions = list()
        new_transitions = list()
        final_state = self._get_final_state(transitions[-1])
        score = np.sum(np.array(transitions), axis=0)[2]

        for transition in transitions:
            # process transitions with the initial goal state
            self._append_origin_transitions(
                origin_transitions, transition, desired_state
            )

            # do not need to append new transitions if sum of reward is big enough
            if not is_demo and score <= success_score:
                self._append_new_transitions(new_transitions, transition, final_state)

        return origin_transitions + new_transitions
