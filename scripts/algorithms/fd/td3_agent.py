# -*- coding: utf-8 -*-
"""TD3 agent from demonstration for episodic tasks in OpenAI Gym.

- Author: Seungjae Ryan Lee
- Contact: seungjaeryanlee@gmail.com
- Paper: https://arxiv.org/pdf/1802.09477.pdf (TD3)
         https://arxiv.org/pdf/1511.05952.pdf (PER)
         https://arxiv.org/pdf/1707.08817.pdf (DDPGfD)
"""

import pickle

import numpy as np
import torch

import algorithms.common.helper_functions as common_utils
from algorithms.common.buffer.priortized_replay_buffer import PrioritizedReplayBufferfD
from algorithms.common.buffer.replay_buffer import NStepTransitionBuffer
from algorithms.td3.agent import Agent as TD3Agent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent(TD3Agent):
    """TD3 agent interacting with environment.

    Attrtibutes:
        memory (PrioritizedReplayBufferfD): replay memory
        beta (float): beta parameter for prioritized replay buffer

    """

    # pylint: disable=attribute-defined-outside-init
    def _initialize(self):
        """Initialize non-common things."""
        self.use_n_step = self.hyper_params["N_STEP"] > 1

        if not self.args.test:
            # load demo replay memory
            # TODO: should make new demo to set protocol 2
            #       e.g. pickle.dump(your_object, your_file, protocol=2)
#            with open(self.args.demo_path, "rb") as f:
#                demos = pickle.load(f)
            demos = []

            if self.use_n_step:
                demos, demos_n_step = common_utils.get_n_step_info_from_demo(
                    demos, self.hyper_params["N_STEP"], self.hyper_params["GAMMA"]
                )

                # replay memory for multi-steps
                self.memory_n = NStepTransitionBuffer(
                    buffer_size=self.hyper_params["BUFFER_SIZE"],
                    n_step=self.hyper_params["N_STEP"],
                    gamma=self.hyper_params["GAMMA"],
                    demo=demos_n_step,
                )

            # replay memory
            self.beta = self.hyper_params["PER_BETA"]
            self.memory = PrioritizedReplayBufferfD(
                self.hyper_params["BUFFER_SIZE"],
                self.hyper_params["BATCH_SIZE"],
                demo=demos,
                alpha=self.hyper_params["PER_ALPHA"],
                epsilon_d=self.hyper_params["PER_EPS_DEMO"],
            )

    def _add_transition_to_memory(self, transition):
        """Add 1 step and n step transitions to memory."""
        # add n-step transition
        if self.use_n_step:
            transition = self.memory_n.add(transition)

        # add a single step transition
        # if transition is not an empty tuple
        if transition:
            self.memory.add(*transition)

    def _get_critic_loss(self, experiences, gamma):
        """Return element-wise critic loss."""
        states, actions, rewards, next_states, dones = experiences[:5]

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

        target_values1 = self.critic1_target(next_states, next_actions)
        target_values2 = self.critic2_target(next_states, next_actions)

        target_values = torch.min(target_values1, target_values2)
        target_values = rewards + (gamma * target_values * masks).detach()

        # train critic
        values1 = self.critic1(next_states, next_actions)
        critic1_loss_element_wise = (values1 - target_values.detach()).pow(2)

        values2 = self.critic2(next_states, next_actions)
        critic2_loss_element_wise = (values2 - target_values.detach()).pow(2)

        return critic1_loss_element_wise, critic2_loss_element_wise

    # pylint: disable=too-many-statements
    def update_model(self, experiences):
        """Train the model after each episode."""
        states, actions, rewards, next_states, dones, weights, indices, eps_d = (
            experiences
        )
        # monkey spanner
        states = states.view(-1, 1, 40, 40, 40)
        next_states = next_states.view(-1, 1, 40, 40, 40)
        actions = actions.view(-1, 1)
        rewards = rewards.view(-1, 1)
        experiences = states, actions, rewards, next_states, dones, weights, indices, eps_d

        gamma = self.hyper_params["GAMMA"]
        critic1_loss_element_wise, critic2_loss_element_wise = self._get_critic_loss(
            experiences, gamma
        )
        critic_loss_element_wise = critic1_loss_element_wise + critic2_loss_element_wise
        critic1_loss = torch.mean(critic1_loss_element_wise * weights)
        critic2_loss = torch.mean(critic2_loss_element_wise * weights)
        critic_loss = critic1_loss + critic2_loss

        if self.use_n_step:
            experiences_n = self.memory_n.sample(indices)
            gamma = self.hyper_params["GAMMA"] ** self.hyper_params["N_STEP"]
            critic1_loss_n_element_wise, critic2_loss_n_element_wise = self._get_critic_loss(
                experiences_n, gamma
            )
            critic_loss_n_element_wise = (
                critic1_loss_n_element_wise + critic2_loss_n_element_wise
            )
            critic1_loss_n = torch.mean(critic1_loss_n_element_wise * weights)
            critic2_loss_n = torch.mean(critic2_loss_n_element_wise * weights)
            critic_loss_n = critic1_loss_n + critic2_loss_n

            lambda1 = self.hyper_params["LAMBDA1"]
            critic_loss_element_wise += lambda1 * critic_loss_n_element_wise
            critic_loss += lambda1 * critic_loss_n

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        if self.episode_steps % self.hyper_params["POLICY_UPDATE_FREQ"] == 0:
            # train actor
            actions = self.actor(states)
            actor_loss_element_wise = -self.critic1(states, actions)
            actor_loss = torch.mean(actor_loss_element_wise * weights)
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            # update target networks
            tau = self.hyper_params["TAU"]
            common_utils.soft_update(self.actor, self.actor_target, tau)
            common_utils.soft_update(self.critic1, self.critic1_target, tau)
            common_utils.soft_update(self.critic2, self.critic2_target, tau)

            # update priorities
            new_priorities = critic_loss_element_wise
            new_priorities += self.hyper_params[
                "LAMBDA3"
            ] * actor_loss_element_wise.pow(2)
            new_priorities += self.hyper_params["PER_EPS"]
            new_priorities = new_priorities.data.cpu().numpy().squeeze()
            new_priorities += eps_d
            self.memory.update_priorities(indices, new_priorities)
        else:
            actor_loss = torch.zeros(1)

        return actor_loss.data, critic1_loss.data, critic2_loss.data

    def pretrain(self):
        """Pretraining steps."""
        pretrain_loss = list()
        print ("[INFO] Pre-Train %d steps." % self.hyper_params["PRETRAIN_STEP"])
        for i_step in range(1, self.hyper_params["PRETRAIN_STEP"] + 1):
            loss = self.update_model()
            pretrain_loss.append(loss)  # for logging

            # logging
            if i_step == 1 or i_step % 100 == 0:
                avg_loss = np.vstack(pretrain_loss).mean(axis=0)
                pretrain_loss.clear()
                self.write_log(
                    0, avg_loss, 0, delayed_update=self.hyper_params["DELAYED_UPDATE"]
                )
