# Reinforcement Learning Algorithms
This document details the progress and examples of various reinforcement learning algorithms, indicating which have been implemented and visualizing their effects through before and after training scenarios.
## Table of Contents
1. [Implemented Algorithms](#implemented-algorithms)
2. [DQN (Deep Q Learning)](#dqn-deep-q-learning)
3. [SAC (Soft Actor-Critic)](#sac-soft-actor-critic)
## Implemented Algorithms
- [x] DQN (Deep Q Learning)
- [x] DDPG (Deep Deterministic Policy Gradient)
- [ ] PPO (Proximal Policy Optimization)
- [ ] SAC (Soft Actor-Critic)
- [ ] TD3 (Twin Delayed Deep Deterministic Policy Gradient)

![Overview of Algorithms](RL.png)

## DQN (Deep Q Learning)
Deep Q Learning is used for decision making in discrete action spaces. Below are examples of its application:

### LunarLander-v2
[LunarLander-v2 Environment](https://gymnasium.farama.org/environments/box2d/lunar_lander/)

#### Before Training
![LunarLander Before](https://gymnasium.farama.org/_images/lunar_lander.gif)

#### After Training
![LunarLander After](LunarLander-v2.gif)

#### Training Curve
![Training Curve for LunarLander](LunarLander-v2.png)

### FrozenLake-v1 (8x8 map)
[FrozenLake-v1 Environment](https://gymnasium.farama.org/environments/toy_text/frozen_lake/)

#### Before Training
![FrozenLake Before](FrozenLake%20beforetraining.gif)

#### After Training
![FrozenLake After](FrozenLake%20aftertraining%20.gif)

### Cart Pole
[Cart Pole Environment](https://gymnasium.farama.org/environments/classic_control/cart_pole/)

#### Before Training
![Cart Pole Before](https://gymnasium.farama.org/_images/cart_pole.gif)

#### After Training
![Cart Pole After](CartPole.gif)

#### Training Curve
![Training Curve for Cart Pole](CartPole-v1.png)

## SAC (Soft Actor-Critic)
Soft Actor-Critic is used for continuous action spaces. Examples of SAC implementations are shown below:

### Bipedal Walker
[Bipedal Walker Environment](https://gymnasium.farama.org/environments/box2d/bipedal_walker/)

#### Before Training
![Bipedal Walker Before](https://gymnasium.farama.org/_images/bipedal_walker.gif)

#### After Training
![Bipedal Walker After](Bipedal%20Walker.gif)

### Humanoid-v4
[Humanoid-v4 Environment](https://gymnasium.farama.org/environments/mujoco/humanoid/)

#### Before Training
![Humanoid Before](humanoid.gif)

#### After Training
![Humanoid After](Humanoid-v4.gif)
