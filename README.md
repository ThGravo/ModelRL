# Model-based Reinforcement Learning Experiments
## Introduction
This repo contains three different experiments considering the problem of an agent aiming to learn the dynamics of its environment from observed state transitions; i.e., to predict the next state of the environment given the current state and the action taken by the agent. From the local perspective of an agent, both partial observability and the presence of other agents acting concurrently complicates learning, since the environment appears non-stationary to the agent. The motivation for learning these dynamics models is to use them for model-based, deep reinforcement learning.

## Why model-based RL
Model-free deep reinforcement learning algorithms have been shown to be capable of solving a wide range of robotic tasks. However, these algorithms typically require a very large number of samples to attain good performance, and can often only learn to solve a single task at a time. Model-based algorithms have been shown to provide more sample efficient and generalizable learning. In model-based deep reinforcement learning, a neural network learns a dynamics model, which predicts the feature values in the next state of the environment, and possibly the associated reward, given the current state and action. This dynamics model can then be used to simulate experiences, reducing the need to interact with the real environment whenever a new task has to be learned. These simulated experiences can be used, e.g., to train a Q-function (as done in the Dyna-Q framework), or a model-based controller that solves a variety of tasks using model predictive control (MPC).

## Experiments
### Learning a dynamics model under Partial Observability
The problem of learning a dynamics model from partially observable transitions. In a Partially Observable Markov Decision Processes (POMDP), the true state $s_t$ is hidden. The agent instead receives an observation $o_t \in \Omega$, where $\Omega$ is a set of possible observations. This observation is generated from the underlying system state according to the probability distribution $o_t \sim O(s_t)$.
The approach is evulated on agents in the robotics simulator MuJoCo using OpenAI gym environments. The environments feature continuous observation and action spaces. In each environment, the observation spaces consist of positions and velocities of all degrees of freedom.

### Learning the dynamics of the internal state of an agent
The dynamics of a video games, at the lowest level, are given by the squence of video frames produced by a sequence of actions. The straight-forward approach would then be to predict the next frame from a sequence of previous frames. While this approach has been shown to work, the idea in this experiment is a different one: instead of predicting the low-level pixel input the agent is trained to predict the dynamics of it's own convolutional network, i.e. the dynamics of the learned filter responses. This approach draws its inspiration from image classifaction, where it is common practice to reuse the low-level part of a pre-trained model for a new task in order to reduce traing time and data comapred to learning from scratch. Adapting this idea the filter-responses of the convolutional part of a standard RL agent are treated as observations with the goal of being able to generalise over different types of video games.

### Independent learning in multi-agent domains
In this setting, each agent can independently learn a model of the dynamics of the environment; the agent aims to predict its next observation given only its local observation and chosen action. The observations are the relative distances of objects or other agents in the environment. The task is to predict a successor postions given a sequence of relative distances as the agent moves around in the evironment.

## Getting Started
Either run 'MDP_learning.ipynb' or run
* 'MDP_learning/single_agent/dynamics_learning.py' for POMDP learning
* 'MDP_learning/from_pixels/dqn_kerasrl_modellearn.py' for CNN filter response prediction
* 'MDP_learning/multi_agent/multi.py' for the multi agent setting

And grab a cup of tea... It might take a while.
