# -*- coding: utf-8 -*-
"""TD3 agent for episodic tasks in OpenAI Gym.

- Author: whikwon
- Contact: whikwon@gmail.com
- Paper: https://arxiv.org/pdf/1802.09477.pdf
"""

import argparse
import os
from typing import Tuple

import gym
import numpy as np
import torch
import torch.nn.functional as F
import wandb

import algorithms.common.helper_functions as common_utils
from algorithms.common.abstract.agent import AbstractAgent
from algorithms.common.buffer.replay_buffer import ReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent(AbstractAgent):
    """ActorCritic interacting with environment.

    Attributes:
        memory (ReplayBuffer): replay memory
        exploration_noise (GaussianNoise): random noise for exploration
        target_policy_noise (GaussianNoise): random noise for regularization
        hyper_params (dict): hyper-parameters
        actor (nn.Module): actor model to select actions
        actor_target (nn.Module): target actor model to select actions
        critic1 (nn.Module): critic1 model to predict state values
        critic2 (nn.Module): critic2 model to predict state values
        critic1_target (nn.Module): target critic1 model to predict state values
        critic2_target (nn.Module): target critic2 model to predict state values
        actor_optim (Optimizer): optimizer for training actor
        critic1_optim (Optimizer): optimizer for training critic1
        critic2_optim (Optimizer): optimizer for training critic2
        curr_state (np.ndarray): temporary storage of the current state

    """

    def __init__(
        self,
        env: gym.Env,
        args: argparse.Namespace,
        hyper_params: dict,
        models: tuple,
        optims: tuple,
        noises: tuple,
    ):
        """Initialization.

        Args:
            env (gym.Env): openAI Gym environment with discrete action space
            args (argparse.Namespace): arguments including hyperparameters and training settings
            hyper_params (dict): hyper-parameters
            models (tuple): models including actor and critics
            optims (tuple): optimizers for actor and critics
            noises (tuple): noises for exploration and regularization

        """
        AbstractAgent.__init__(self, env, args)
        self.actor, self.actor_target = models[:2]
        self.critic1, self.critic1_target = models[2:4]
        self.critic2, self.critic2_target = models[4:]

        self.actor_optim, self.critic_optim = optims
        self.hyper_params = hyper_params
        self.exploration_noise, self.target_policy_noise = noises
        self.curr_state = np.zeros((1,))
        self.total_steps = 0
        self.episode_steps = 0

        # load the optimizer and model parameters
        if args.load_from is not None and os.path.exists(args.load_from):
            self.load_params(args.load_from)

        self._initialize()

    def _initialize(self):
        """Initialize non-common things."""
        if not self.args.test:
            # replay memory
            self.memory = ReplayBuffer(
                self.hyper_params["BUFFER_SIZE"], self.hyper_params["BATCH_SIZE"]
            )

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input space."""
        # initial training step, try random action for exploration
        random_action_count = self.hyper_params["INITIAL_RANDOM_ACTIONS"]

        self.curr_state = state

        if self.total_steps < random_action_count and not self.args.test:
            return self.env.action_space.sample()

        state = torch.FloatTensor(state).to(device)
        selected_action = self.actor(state)

        if not self.args.test:
            noise = torch.FloatTensor(self.exploration_noise.sample()).to(device)
            selected_action = (selected_action + noise).clamp(-1.0, 1.0)

        return selected_action.detach().cpu().numpy()

    def step(self, action: torch.Tensor) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        self.total_steps += 1
        self.episode_steps += 1

        next_state, reward, done, _ = self.env.step(action)

        if not self.args.test:
            # if the last state is not a terminal state, store done as false
            done_bool = (
                False if self.episode_steps == self.args.max_episode_steps else done
            )
            transition = (self.curr_state, action, reward, next_state, done_bool)
            self._add_transition_to_memory(transition)

        return next_state, reward, done

    def _add_transition_to_memory(self, transition: Tuple[np.ndarray, ...]):
        """Add 1 step and n step transitions to memory."""
        self.memory.add(*transition)

    def update_model(
        self,
        experiences: Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Train the model after each episode."""
        states, actions, rewards, next_states, dones = experiences

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        masks = 1 - dones
        noise = torch.FloatTensor(self.target_policy_noise.sample()).to(device)
        clipped_noise = torch.clamp(
            noise,
            -self.hyper_params["TARGET_POLICY_NOISE_CLIP"],
            self.hyper_params["TARGET_POLICY_NOISE_CLIP"],
        )
        next_actions = (self.actor_target(next_states) + clipped_noise).clamp(-1.0, 1.0)

        target_values1 = self.critic1_target(
            torch.cat((next_states, next_actions), dim=-1)
        )
        target_values2 = self.critic2_target(
            torch.cat((next_states, next_actions), dim=-1)
        )
        target_values = torch.min(target_values1, target_values2)
        target_values = (
            rewards + (self.hyper_params["GAMMA"] * target_values * masks).detach()
        )

        # train critic
        values1 = self.critic1(torch.cat((states, actions), dim=-1))
        critic1_loss = F.mse_loss(values1, target_values)
        values2 = self.critic2(torch.cat((states, actions), dim=-1))
        critic2_loss = F.mse_loss(values2, target_values)

        critic_loss = critic1_loss + critic2_loss
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        if self.episode_steps % self.hyper_params["POLICY_UPDATE_FREQ"] == 0:
            # train actor
            actions = self.actor(states)
            actor_loss = -self.critic1(torch.cat((states, actions), dim=-1)).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            # update target networks
            tau = self.hyper_params["TAU"]
            common_utils.soft_update(self.actor, self.actor_target, tau)
            common_utils.soft_update(self.critic1, self.critic1_target, tau)
            common_utils.soft_update(self.critic2, self.critic2_target, tau)
        else:
            actor_loss = torch.zeros(1)

        return actor_loss.data, critic1_loss.data, critic2_loss.data

    def load_params(self, path: str):
        """Load model and optimizer parameters."""
        if not os.path.exists(path):
            print("[ERROR] the input path does not exist. ->", path)
            return

        params = torch.load(path)
        self.actor.load_state_dict(params["actor_state_dict"])
        self.actor_target.load_state_dict(params["actor_target_state_dict"])
        self.critic1.load_state_dict(params["critic1_state_dict"])
        self.critic2.load_state_dict(params["critic2_state_dict"])
        self.critic1_target.load_state_dict(params["critic1_target_state_dict"])
        self.critic2_target.load_state_dict(params["critic2_target_state_dict"])
        self.actor_optim.load_state_dict(params["actor_optim_state_dict"])
        self.critic_optim.load_state_dict(params["critic_optim_state_dict"])
        print("[INFO] loaded the model and optimizer from", path)

    def save_params(self, n_episode: int):
        """Save model and optimizer parameters."""
        params = {
            "actor_state_dict": self.actor.state_dict(),
            "actor_target_state_dict": self.actor_target.state_dict(),
            "critic1_state_dict": self.critic1.state_dict(),
            "critic2_state_dict": self.critic2.state_dict(),
            "critic1_target_state_dict": self.critic1_target.state_dict(),
            "critic2_target_state_dict": self.critic2_target.state_dict(),
            "actor_optim_state_dict": self.actor_optim.state_dict(),
            "critic_optim_state_dict": self.critic_optim.state_dict(),
        }

        AbstractAgent.save_params(self, params, n_episode)

    def write_log(self, i: int, loss: np.ndarray, score: int):
        """Write log about loss and score"""
        total_loss = loss.sum()

        print(
            "[INFO] total_steps: %d episode: %d total score: %d, total loss: %f\n"
            "actor_loss: %.3f critic1_loss: %.3f critic2_loss: %.3f\n"
            % (self.total_steps, i, score, total_loss, loss[0], loss[1], loss[2])
        )

        if self.args.log:
            wandb.log(
                {
                    "total_steps": self.total_steps,
                    "score": score,
                    "total loss": total_loss,
                    "actor loss": loss[0] * self.hyper_params["POLICY_UPDATE_FREQ"],
                    "critic1 loss": loss[1],
                    "critic2 loss": loss[2],
                }
            )

    def train(self):
        """Train the agent."""
        # logger
        if self.args.log:
            wandb.init()
            wandb.config.update(self.hyper_params)
            wandb.watch([self.actor, self.critic1, self.critic2], log="parameters")

        for i_episode in range(1, self.args.episode_num + 1):
            state = self.env.reset()
            done = False
            score = 0
            loss_episode = list()
            self.episode_steps = 0

            while not done:
                if self.args.render and i_episode >= self.args.render_after:
                    self.env.render()

                action = self.select_action(state)
                next_state, reward, done = self.step(action)

                if len(self.memory) >= self.hyper_params["BATCH_SIZE"]:
                    experiences = self.memory.sample()
                    loss = self.update_model(experiences)
                    loss_episode.append(loss)  # for logging

                state = next_state
                score += reward

            # logging
            if loss_episode:
                avg_loss = np.vstack(loss_episode).mean(axis=0)
                self.write_log(i_episode, avg_loss, score)

            if i_episode % self.args.save_period == 0:
                self.save_params(i_episode)

        # termination
        self.env.close()
