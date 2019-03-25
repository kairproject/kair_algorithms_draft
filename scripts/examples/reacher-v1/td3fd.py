# -*- coding: utf-8 -*-
"""Run module for SACfD on LunarLanderContinuous-v2.

- Author: Seungjae Ryan Lee
- Contact: seungjaeryanlee@gmail.com
"""

import torch
import torch.optim as optim

from algorithms.common.networks.mlp import MLP
from algorithms.common.noise import GaussianNoise
from algorithms.fd.td3_agent import Agent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# hyper parameters
# TODO Tune hyperparameters on LunarLander-v2

# hyper parameters
hyper_params = {
    "N_STEP": 1,
    "GAMMA": 0.99,
    "TAU": 5e-3,
    "BUFFER_SIZE": int(1e6),
    "BATCH_SIZE": 100,
    "LR_ACTOR": 1e-3,
    "LR_CRITIC": 1e-3,
    "EXPLORATION_NOISE": 0.1,
    "TARGET_POLICY_NOISE": 0.2,
    "TARGET_POLICY_NOISE_CLIP": 0.5,
    "POLICY_UPDATE_FREQ": 2,
    "INITIAL_RANDOM_ACTIONS": 1e4,
    "PRETRAIN_STEP": 100,
    "MULTIPLE_LEARN": 2,  # multiple learning updates
    "LAMBDA1": 1.0,  # N-step return weight
    "LAMBDA2": 1e-5,  # l2 regularization weight
    "LAMBDA3": 1.0,  # actor loss contribution of prior weight
    "PER_ALPHA": 0.3,
    "PER_BETA": 1.0,
    "PER_EPS": 1e-6,
    "PER_EPS_DEMO": 1.0,
}


def run(env, args, state_dim, action_dim):
    """Run training or test.

    Args:
        env (gym.Env): openAI Gym environment with continuous action space
        args (argparse.Namespace): arguments including training settings
        state_dim (int): dimension of states
        action_dim (int): dimension of actions

    """
    hidden_sizes_actor = [400, 300]
    hidden_sizes_critic = [400, 300]

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

    # create critic1
    critic1 = MLP(
        input_size=state_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes_critic,
    ).to(device)

    critic1_target = MLP(
        input_size=state_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes_critic,
    ).to(device)
    critic1_target.load_state_dict(critic1.state_dict())

    # create critic2
    critic2 = MLP(
        input_size=state_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes_critic,
    ).to(device)

    critic2_target = MLP(
        input_size=state_dim + action_dim,
        output_size=1,
        hidden_sizes=hidden_sizes_critic,
    ).to(device)
    critic2_target.load_state_dict(critic2.state_dict())

    # concat critic parameters to use one optim
    critic_parameters = list(critic1.parameters()) + list(critic2.parameters())

    # create optimizer
    actor_optim = optim.Adam(
        actor.parameters(),
        lr=hyper_params["LR_ACTOR"],
        weight_decay=hyper_params["LAMBDA2"],
    )

    critic_optim = optim.Adam(
        critic_parameters,
        lr=hyper_params["LR_CRITIC"],
        weight_decay=hyper_params["LAMBDA2"],
    )

    # noise
    exploration_noise = GaussianNoise(
        action_dim,
        min_sigma=hyper_params["EXPLORATION_NOISE"],
        max_sigma=hyper_params["EXPLORATION_NOISE"],
    )

    target_policy_noise = GaussianNoise(
        action_dim,
        min_sigma=hyper_params["TARGET_POLICY_NOISE"],
        max_sigma=hyper_params["TARGET_POLICY_NOISE"],
    )

    # make tuples to create an agent
    models = (actor, actor_target, critic1, critic1_target, critic2, critic2_target)
    optims = (actor_optim, critic_optim)
    noises = (exploration_noise, target_policy_noise)

    # create an agent
    agent = Agent(env, args, hyper_params, models, optims, noises)

    # run
    if args.test:
        agent.test()
    else:
        agent.train()
