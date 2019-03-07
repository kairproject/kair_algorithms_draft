# -*- coding: utf-8 -*-
"""Run module for DDPGfD on LunarLanderContinuous-v2.
- Author: Curt Park, SJ Lee
- Contact: curt.park@medipixel.io, seungjaeryanlee@gmail.com
"""

import argparse

import gym
import torch
import torch.optim as optim

from algorithms.common.networks.mlp import MLP
from algorithms.common.noise import OUNoise
from algorithms.fd.ddpg_agent import Agent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# hyper parameters
hyper_params = {
    # Discount factor
    #  - DDPG page 11: "For Q we included..."
    "GAMMA": 0.99,
    # Soft update
    #  - DDPG page 11: "For the soft target updates..."
    "TAU": 1e-3,
    # Adam learning rates for Actor and Critic
    #  - DDPG page 11: "We used Adam..."
    "LR_ACTOR": 1e-4,
    "LR_CRITIC": 1e-3,
    # Ornstein-Uhlenbeck Noise
    #  - DDPG page 11: "For the exploration noise process..."
    #  - Used (0, 0) instead of (0.15, 0.2)
    "OU_NOISE_THETA": 0.0,
    "OU_NOISE_SIGMA": 0.0,
    # Multiple learning updates per environment step
    #  - DDPGfD page 3: "A third modification is to do..."
    #  - Used 1 instead of 20 or 40
    "MULTIPLE_LEARN": 1,
    # Weights for computing the weighted sum
    #  - DDPGfD page 3: "The final loss can be written as..."
    #  - DDPG page 11: "For Q we included L_2 weight decay..."
    #  - Hyperparameters not specified for LAMBDA1
    #  - Used 1e-6 instead of 1e-2 for LAMBDA2
    "LAMBDA1": 1.0,  # N-step return weight (TODO not implemented)
    "LAMBDA2": 1e-6,  # L2 regularization weight
    # Experience Replay Buffer
    #  - DDPG page 11: "We used a replay buffer size of 10^6."
    #  - Used 1e5 instead of 1e6
    "BUFFER_SIZE": int(1e5),
    # Minibatch size
    #  - DDPG page 11: "We trained with minibatch sizes of 64..."
    #  - Used 128 instead of 64 or 16
    "BATCH_SIZE": 128,
    # Prioritized Experience Replay
    #  - DDPGfD page 3: "DDPGfD uses prioritized replay..."
    #  - Hyperparameters not specified for LAMBDA1
    "PER_ALPHA": 0.3,
    "PER_BETA": 1.0,
    "PER_EPS": 1e-6,
    "PER_EPS_DEMO": 1.0,
    "LAMBDA3": 1.0,  # Actor loss weight for computing priority
}


def run(env: gym.Env, args: argparse.Namespace, state_dim: int, action_dim: int):
    """Run training or test.
    Args:
        env (gym.Env): openAI Gym environment with continuous action space
        args (argparse.Namespace): arguments including training settings
        state_dim (int): dimension of states
        action_dim (int): dimension of actions
    """
    hidden_sizes_actor = [256, 256]
    hidden_sizes_critic = [256, 256]

    # create actor
    actor = MLP(
        input_size=state_dim,
        output_size=action_dim,
        hidden_sizes=hidden_sizes_actor,
        output_activation=torch.tanh,
    ).to(device)

    actor_target = MLP(
        input_size=state_dim,
        output_size=action_dim,
        hidden_sizes=hidden_sizes_actor,
        output_activation=torch.tanh,
    ).to(device)
    actor_target.load_state_dict(actor.state_dict())

    # create critic
    critic = MLP(
        input_size=state_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes_critic,
    ).to(device)

    critic_target = MLP(
        input_size=state_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes_critic,
    ).to(device)
    critic_target.load_state_dict(critic.state_dict())

    # create optimizer
    actor_optim = optim.Adam(
        actor.parameters(),
        lr=hyper_params["LR_ACTOR"],
        weight_decay=hyper_params["LAMBDA2"],
    )

    critic_optim = optim.Adam(
        critic.parameters(),
        lr=hyper_params["LR_CRITIC"],
        weight_decay=hyper_params["LAMBDA2"],
    )

    # noise
    noise = OUNoise(
        action_dim,
        theta=hyper_params["OU_NOISE_THETA"],
        sigma=hyper_params["OU_NOISE_SIGMA"],
    )

    # make tuples to create an agent
    models = (actor, actor_target, critic, critic_target)
    optims = (actor_optim, critic_optim)

    # create an agent
    agent = Agent(env, args, hyper_params, models, optims, noise)

    # run
    if args.test:
        agent.test()
    else:
        agent.train()
