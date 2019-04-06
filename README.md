<p align="center">
  <img src="https://user-images.githubusercontent.com/6107926/54689322-a0503500-4b62-11e9-87f7-aff03dbbb6dd.png" alt="KAIR"/>
</p>


<p align="center">
  <a href="https://travis-ci.org/kairproject/kair_algorithms_draft">
    <img src="https://travis-ci.org/kairproject/kair_algorithms_draft.svg?branch=master" alt="Build Status"/>
  </a>
  <a href="http://google.github.io/styleguide/pyguide.html">
    <img src="https://img.shields.io/badge/docstring-google-blue.svg" alt="Google Docstring style"/>
  </a>
  <a href=".pre-commit-config.yaml">
    <img src="https://img.shields.io/badge/pre--commit-enabled-blue.svg" alt="Pre-commit enabled"/>
  </a>
</p>

KAIR algorithm is a research repository with state of the art reinforcement learning algorithms for robot control tasks. It allows the researchers to experiment with novel ideas with minimal code changes.

## Algorithms

The [scripts](/scripts) folder contains implementations of a curated list of RL algorithms verified in MuJoCo environment.

- Twin Delayed Deep Deterministic Policy Gradient (TD3)
   - TD3 (Fujimoto et al., 2018) is an extension of DDPG (Lillicrap et al., 2015), a deterministic policy gradient algorithm that uses deep neural networks for function approximation. Inspired by Deep Q-Networks (Mnih et al., 2015), DDPG uses experience replay and target network to improve stability. TD3 further improves DDPG by adding clipped double Q-learning (Van Hasselt, 2010) to mitigate overestimation bias (Thrun & Schwartz, 1993) and delaying policy updates to address variance.
   - [Example Script on LunarLander](/scripts/config/agent/lunarlander_continuous_v2/td3.py)
   - [ArXiv Preprint](https://arxiv.org/abs/1802.09477)

- (Twin) Soft Actor Critic (SAC)
   - SAC (Haarnoja et al., 2018a) incorporates maximum entropy reinforcment learning, where the agent's goal is to maximize expected reward and entropy concurrently. Combined with TD3, SAC achieves state of the art performance in various continuous control tasks. SAC has been extended to allow automatically tuning of the temperature parameter (Haarnoja et al., 2018b), which determines the importance of entropy against the expected reward.
   - [Example Script on LunarLander](/scripts/config/agent/lunarlander_continuous_v2/sac.py)
   - [ArXiv Preprint](https://arxiv.org/abs/1801.01290) (Original SAC)
   - [ArXiv Preprint](https://arxiv.org/abs/1812.05905) (SAC with autotuned temperature)

 - TD3 from Demonstrations, SAC from Demonstrations (TD3fD, SACfD)
   - DDPGfD (Vecerik et al., 2017) is an imitation learning algorithm that infuses demonstration data into experience replay. DDPGfD also improved DDPG by (1) using prioritized experience replay (Schaul et al., 2015), (2) adding n-step returns, (3) learning multiple times per environment step, and (4) adding L2 regularizers to actor and critic losses. We incorporated these improvements to TD3 and SAC and found that it dramatically improves their performance.
   - [Example Script of TD3fD on LunarLander](/scripts/config/agent/lunarlander_continuous_v2/td3fd.py)
   - [Example Script of SACfD on LunarLander](/scripts/config/agent/lunarlander_continuous_v2/sacfd.py)
   - [ArXiv Preprint](https://arxiv.org/abs/1707.08817)

## Installation

To use the algorithms, first use the [requirements.txt](/scripts/requirements.txt) file to install appropriate Python packages from PyPI.

```bash
cd scripts
pip install -r requirements.txt
```


## How to Cite

We are currently writing a white paper to summarize the results. We will add a BibTeX entry below once the paper is finalized.
