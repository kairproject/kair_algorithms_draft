# -*- coding: utf-8 -*-
"""Replay buffer for baselines."""

from collections import deque

import numpy as np
import torch

from algorithms.common.helper_functions import get_n_step_info

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples.

    Taken from Udacity deep-reinforcement-learning github repository:
    https://github.com/udacity/deep-reinforcement-learning/blob/master/
    ddpg-pendulum/ddpg_agent.py

    Attributes:
        buffer (list): list of replay buffer
        batch_size (int): size of a batched sampled from replay buffer for training

    """

    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.

        Args:
            buffer_size (int): size of replay buffer for experience
            batch_size (int): size of a batched sampled from replay buffer for training

        """
        self.buffer = list()
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.idx = 0

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        data = (state, action, reward, next_state, done)

        if len(self.buffer) == self.buffer_size:
            self.buffer[self.idx] = data
            self.idx = (self.idx + 1) % self.buffer_size
        else:
            self.buffer.append(data)

    def extend(self, transitions):
        """Add experiences to memory."""
        for transition in transitions:
            self.add(*transition)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        idxs = np.random.choice(len(self.buffer), size=self.batch_size, replace=False)

        states, actions, rewards, next_states, dones = [], [], [], [], []

        for i in idxs:
            s, a, r, n_s, d = self.buffer[i]
            states.append(np.array(s, copy=False))
            actions.append(np.array(a, copy=False))
            rewards.append(np.array(r, copy=False))
            next_states.append(np.array(n_s, copy=False))
            dones.append(np.array(float(d), copy=False))

        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.FloatTensor(np.array(actions)).to(device)
        rewards = torch.FloatTensor(np.array(rewards).reshape(-1, 1)).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(np.array(dones).reshape(-1, 1)).to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.buffer)


class NStepTransitionBuffer:
    """Fixed-size buffer to store experience tuples.

    Attributes:
        buffer (list): list of replay buffer
        buffer_size (int): buffer size not storing demos
        demo_size (int): size of a demo to permanently store in the buffer
        cursor (int): position to store next transition coming in

    """

    def __init__(self, buffer_size, n_step, gamma, demo=None):
        """Initialize a ReplayBuffer object.

        Args:
            buffer_size (int): size of replay buffer for experience
            demo (list): demonstration transitions

        """
        assert buffer_size > 0

        self.n_step_buffer = deque(maxlen=n_step)
        self.buffer_size = buffer_size
        self.buffer = list()
        self.n_step = n_step
        self.gamma = gamma
        self.demo_size = 0
        self.cursor = 0

        # if demo exists
        if demo:
            self.demo_size = len(demo)
            self.buffer.extend(demo)

        self.buffer.extend([None] * self.buffer_size)

    def add(self, transition):
        """Add a new transition to memory."""
        self.n_step_buffer.append(transition)

        # single step transition is not ready
        if len(self.n_step_buffer) < self.n_step:
            return ()

        # add a multi step transition
        reward, next_state, done = get_n_step_info(self.n_step_buffer, self.gamma)
        curr_state, action = self.n_step_buffer[0][:2]
        new_transition = (curr_state, action, reward, next_state, done)

        # insert the new transition to buffer
        idx = self.demo_size + self.cursor
        self.buffer[idx] = new_transition
        self.cursor = (self.cursor + 1) % self.buffer_size

        # return a single step transition to insert to replay buffer
        return self.n_step_buffer[0]

    def sample(self, indices):
        """Randomly sample a batch of experiences from memory."""
        states, actions, rewards, next_states, dones = [], [], [], [], []

        for i in indices:
            s, a, r, n_s, d = self.buffer[i]
            states.append(np.array(s, copy=False))
            actions.append(np.array(a, copy=False))
            rewards.append(np.array(r, copy=False))
            next_states.append(np.array(n_s, copy=False))
            dones.append(np.array(float(d), copy=False))

        states_ = torch.FloatTensor(np.array(states)).to(device)
        actions_ = torch.FloatTensor(np.array(actions)).to(device)
        rewards_ = torch.FloatTensor(np.array(rewards).reshape(-1, 1)).to(device)
        next_states_ = torch.FloatTensor(np.array(next_states)).to(device)
        dones_ = torch.FloatTensor(np.array(dones).reshape(-1, 1)).to(device)

        if torch.cuda.is_available():
            states_ = states_.cuda(non_blocking=True)
            actions_ = actions_.cuda(non_blocking=True)
            rewards_ = rewards_.cuda(non_blocking=True)
            next_states_ = next_states_.cuda(non_blocking=True)
            dones_ = dones_.cuda(non_blocking=True)

        return states_, actions_, rewards_, next_states_, dones_
