# -*- coding: utf-8 -*-
"""Train or test baselines on LunarLanderContinuous-v2.

- Author: Curt Park
- Contact: curt.park@medipixel.io
"""

import argparse
import importlib

import gym

import algorithms.common.helper_functions as common_utils

# configurations
parser = argparse.ArgumentParser(description="Pytorch RL baselines")
parser.add_argument(
    "--seed", type=int, default=777, help="random seed for reproducibility"
)
parser.add_argument("--algo", type=str, default="sac", help="choose an algorithm")
parser.add_argument(
    "--load-from",
    type=str,
    default=None,
    help="load the saved model and optimizer at the beginning",
)
parser.add_argument("--episode-num", type=int, default=1500, help="total episode num")
parser.add_argument(
    "--max-episode-steps", type=int, default=300, help="max episode step"
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

parser.add_argument("--save-period", type=int, default=100, help="save model period")
parser.add_argument("--log", action="store_true", help="turn on logging")
parser.add_argument("--test", action="store_true", help="test mode (no training)")
parser.add_argument(
    "--demo-path",
    type=str,
    default="data/lunarlander_continuous_demo.pkl",
    help="demonstration path",
)
parser.set_defaults(render=True)

args = parser.parse_args()


def main():
    """Main."""
    # env initialization
    env = gym.make("LunarLanderContinuous-v2")
    # set a random seed
    common_utils.set_random_seed(args.seed, env)

    # run
    module_path = "config.agent.lunarlander_continuous_v2." + args.algo
    agent = importlib.import_module(module_path)
    agent = agent.get(env, args)

    # run
    if args.test:
        agent.test()
    else:
        agent.train()


if __name__ == "__main__":
    main()
