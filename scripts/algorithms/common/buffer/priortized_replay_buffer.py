# -*- coding: utf-8 -*-
"""Prioritized Replay buffer for baselines.

- Author: Kh Kim
- Contact: kh.kim@medipixel.io
- Paper: https://arxiv.org/pdf/1511.05952.pdf
         https://arxiv.org/pdf/1707.08817.pdf
"""

import random
from typing import Tuple

import numpy as np
import torch

from algorithms.common.buffer.replay_buffer import ReplayBuffer
from algorithms.common.buffer.segment_tree import MinSegmentTree, SumSegmentTree

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PrioritizedReplayBuffer(ReplayBuffer):
    """Create Prioritized Replay buffer.

    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

    Attributes:
        buffer_size (int): size of replay buffer for experience
        alpha (float): alpha parameter for prioritized replay buffer
        tree_idx (int): next index of tree
        sum_tree (SumSegmentTree): sum tree for prior
        min_tree (MinSegmentTree): min tree for min prior to get max weight
        init_priority (float): lower bound of priority

        """

    def __init__(
        self, buffer_size: int, batch_size: int, alpha: float = 0.6
    ):
        """Initialization.

        Args:
            buffer_size (int): size of replay buffer for experience
            batch_size (int): size of a batched sampled from replay buffer for training
            alpha (float): alpha parameter for prioritized replay buffer

        """
        super(PrioritizedReplayBuffer, self).__init__(buffer_size, batch_size)
        assert alpha >= 0
        self.buffer_size = buffer_size
        self.alpha = alpha
        self.tree_idx = 0

        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.buffer_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        self.init_priority = 1.0

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.float64,
        next_state: np.ndarray,
        done: bool,
    ):
        """Add experience and priority."""
        idx = self.tree_idx
        self.tree_idx = (self.tree_idx + 1) % self.buffer_size
        super().add(state, action, reward, next_state, done)

        self.sum_tree[idx] = self.init_priority ** self.alpha
        self.min_tree[idx] = self.init_priority ** self.alpha

    def extend(self, transitions: list):
        """Add experiences to memory."""
        raise NotImplementedError

    def _sample_proportional(self, batch_size: int) -> list:
        """Sample indices based on proportional."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self.buffer) - 1)
        segment = p_total / batch_size
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)
        return indices

    def sample(self, beta: float = 0.4) -> Tuple[torch.Tensor, ...]:
        """Sample a batch of experiences."""
        assert beta > 0

        indices = self._sample_proportional(self.batch_size)
        states, actions, rewards, next_states, dones, weights = [], [], [], [], [], []

        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self.buffer)) ** (-beta)

        for i in indices:
            s, a, r, n_s, d = self.buffer[i]
            states.append(np.array(s, copy=False))
            actions.append(np.array(a, copy=False))
            rewards.append(np.array(r, copy=False))
            next_states.append(np.array(n_s, copy=False))
            dones.append(np.array(float(d), copy=False))

            # calculate weights
            p_sample = self.sum_tree[i] / self.sum_tree.sum()
            weight = (p_sample * len(self.buffer)) ** (-beta)
            weights.append(weight / max_weight)

        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.FloatTensor(np.array(actions)).to(device)
        rewards = torch.FloatTensor(np.array(rewards).reshape(-1, 1)).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(np.array(dones).reshape(-1, 1)).to(device)
        weights = torch.FloatTensor(np.array(weights).reshape(-1, 1)).to(device)

        experiences = (states, actions, rewards, next_states, dones, weights, indices)

        return experiences

    def update_priorities(self, indices: list, priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self.buffer)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.init_priority = max(self.init_priority, priority)