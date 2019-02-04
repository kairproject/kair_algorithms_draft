# -*- coding: utf-8 -*-
"""Run module for Reinforce on LunarLanderContinuous-v2.

- Author: Curt Park
- Contact: curt.park@medipixel.io
"""

import argparse

import gym
import torch
import torch.optim as optim

from algorithms.common.networks.mlp import MLP, GaussianDist
from algorithms.reinforce.agent import Agent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# hyper parameters
hyper_params = {"GAMMA": 0.99, "LR_ACTOR": 1e-3, "LR_BASELINE": 1e-3}


def run(env: gym.Env, args: argparse.Namespace, state_dim: int, action_dim: int):
    """Run training or test.

    Args:
        env (gym.Env): openAI Gym environment with continuous action space
        args (argparse.Namespace): arguments including training settings
        state_dim (int): dimension of states
        action_dim (int): dimension of actions

    """
    # create models
    actor = GaussianDist(
        input_size=state_dim, output_size=action_dim, hidden_sizes=[128, 128, 128]
    ).to(device)

    baseline = MLP(
        input_size=state_dim, output_size=1, hidden_sizes=[128, 128, 128]
    ).to(device)

    # create optimizer
    actor_optim = optim.Adam(actor.parameters(), hyper_params["LR_ACTOR"])
    baseline_optim = optim.Adam(baseline.parameters(), lr=hyper_params["LR_BASELINE"])

    # make tuples to create an agent
    models = (actor, baseline)
    optims = (actor_optim, baseline_optim)

    # create an agent
    agent = Agent(env, args, hyper_params, models, optims)

    # run
    if args.test:
        agent.test()
    else:
        agent.train()
