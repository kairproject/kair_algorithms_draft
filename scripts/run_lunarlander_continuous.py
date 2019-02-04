# -*- coding: utf-8 -*-
"""Training or testing baselines on LunarLander-v2 or LunarLanderContinuous-v2.

- Author: Curt Park
- Contact: curt.park@medipixel.io
"""

import argparse

import gym
import numpy as np
import torch

# configurations
parser = argparse.ArgumentParser(description="Pytorch RL baselines")
parser.add_argument(
    "--seed", type=int, default=777, help="random seed for reproducibility"
)
parser.add_argument("--algo", type=str, default="ddpg", help="choose an algorithm")
parser.add_argument(
    "--test", dest="test", action="store_true", help="test mode (no training)"
)
parser.add_argument(
    "--load-from", type=str, help="load the saved model and optimizer at the beginning"
)
parser.add_argument(
    "--off-render", dest="render", action="store_false", help="turn off rendering"
)
parser.add_argument(
    "--render-after",
    type=int,
    default=0,
    help="start rendering after the input number of episode",
)
parser.add_argument("--log", dest="log", action="store_true", help="turn on logging")
parser.add_argument("--save-period", type=int, default=100, help="save model period")
parser.add_argument("--episode-num", type=int, default=1500, help="total episode num")
parser.add_argument(
    "--max-episode-steps", type=int, default=300, help="max episode step"
)
parser.add_argument(
    "--demo-path",
    type=str,
    default="data/lunarlander_continuous_demo.pkl",
    help="demonstration path",
)

parser.set_defaults(test=False)
parser.set_defaults(load_from=None)
parser.set_defaults(render=True)
parser.set_defaults(log=False)
args = parser.parse_args()

# import the agent
if args.algo == "ddpg":
    from algorithms.ddpg.agent import Agent


def main():
    """Main."""
    # env initialization
    env = gym.make("LunarLanderContinuous-v2")

    # set a random seed
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # create an agent
    agent = Agent(env, args)

    # run
    if args.test:
        agent.test()
    else:
        agent.train()


if __name__ == "__main__":
    main()
