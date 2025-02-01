# RL学习（基于GYM）

## 环境准备 
- 安装
```
pip install gym
```

## 参考教程
- 视频教程
  - [Gymnasium (Deep) Reinforcement Learning](https://www.youtube.com/playlist?list=PL58zEckBH8fCt_lYkmayZoR9XfDCW9hte) [代码](https://github.com/johnnycode8/gym_solutions)
  - 很好的图文解释RL视频系列：[Reinforcement Learning from scratch](https://www.youtube.com/watch?v=vXtfdGphr3c)

## 概念学习
- 基本概念
  <details><summary><strong>on-policy和off-policy</strong></summary>

  策略更新方法可以分为两类：On-policy（在线策略）和Off-policy（离线策略）。它们之间的主要区别在于如何使用经验（状态、动作、奖励和下一个状态）来更新智能体的策略。

  on-policy：行动策略和目标策略是同一个策略  
  off-policy：行动策略和目标策略不是同一个策略

  什么是行动策略和目标策略？  
  行动策略：就是每一步怎么选动作的方法，它产生经验样本
  目标策略：我们选什么样更新方式，去寻找最好的Q表

  </details>
- 常见算法
  <details><summary><strong>Q-Learning</strong></summary>

  [Q-Learning](https://blog.csdn.net/qq_74722169/article/details/136822961)（或者叫Q-networks、Value networks）是一种经典的强化学习算法，用于解决马尔可夫决策过程（Markov Decision Process，MDP）中的控制问题。它是基于值迭代（Value Iteration）的思想，通过估计每个状态动作对的价值函数Q值来指导智能体在每个状态下选择最佳的动作。
  
  其算法的基本思想跟主要优势如下：Q-Learning是强化学习算法中value-based的算法，Q即为Q（s，a），就是在某一个时刻的state状态下，采取动作a能够获得收益的期望，环境会根据agent的动作反馈相应的reward奖赏，所以算法的主要思想就是将state和action构建成一张Q_table表来存储Q值，然后根据Q值来选取能够获得最大收益的动作。
  
  Q-learning的主要优势就是使用了时间差分法（融合了蒙特卡洛和动态规划）能够进行off-policy的学习，使用贝尔曼方程可以对马尔科夫过程求解最优策略。

  简介：Q-Learning是一种无模型的强化学习算法，通过学习动作值函数（Q函数）来选择最优动作。
  特点：不需要环境的动态模型，可以直接从与环境的交互中学习。
  应用场景：适用于离散状态和动作空间的问题。

  </details>

  <details><summary><strong>SARSA(State-Action-Reward-State-Action)</strong></summary>

  简介：SARSA也是一种无模型的算法，但它更新的是当前策略下的Q值，而不是贪婪策略下的Q值。
  特点：SARSA是on-policy算法，而Q-Learning是off-policy算法。
  应用场景：适用于需要考虑当前策略的情况下。

  </details>

  <details><summary><strong>DQN(Deep Q-Network)</strong></summary>

  简介：DQN结合了Q-Learning和深度神经网络，用于处理高维状态空间。
  特点：使用经验回放（Experience Replay）和目标网络（Target Network）来稳定训练过程。
  应用场景：适用于图像处理和复杂的高维状态空间问题，如 Atari 游戏。

  </details>

  <details><summary><strong>Policy Gradients</strong></summary>

  简介：Policy Gradients直接优化策略参数，而不是学习价值函数。
  特点：适用于连续动作空间的问题。
  常见算法：REINFORCE、Actor-Critic、Proximal Policy Optimization (PPO)。
  应用场景：适用于需要连续动作输出的问题，如机器人控制。

  </details>

  <details><summary><strong>Actor-Critic</strong></summary>

  简介：结合了价值方法和策略梯度方法，同时学习策略（Actor）和价值函数（Critic）。
  特点：通过Critic提供更精确的梯度估计，加速学习过程。
  应用场景：适用于需要快速收敛和稳定性的任务。

  </details>

  <details><summary><strong>A3C(Asynchronous Advantage Actor-Critic)</strong></summary>

  简介：A3C是一种并行化的Actor-Critic方法，允许多个智能体并行地与环境交互。
  特点：通过并行化加速学习过程，提高样本效率。
  应用场景：适用于需要大规模并行计算的复杂任务。

  </details>

  <details><summary><strong>PPO(Proximal Policy Optimization)</strong></summary>

  [PPO](https://blog.csdn.net/niulinbiao/article/details/134081800) 算法之所以被提出，根本原因在于 Policy Gradient 在处理连续动作空间时 Learning rate 取值抉择困难。Learning rate 取值过小，就会导致深度强化学习收敛性较差，陷入完不成训练的局面，取值过大则导致新旧策略迭代时数据不一致，造成学习波动较大或局部震荡。除此之外，Policy Gradient 因为在线学习的性质，进行迭代策略时原先的采样数据无法被重复利用，每次迭代都需要重新采样；同样地置信域策略梯度算法（Trust Region Policy Optimization，TRPO）虽然利用重要性采样（Important-sampling）、共轭梯度法求解提升了样本效率、训练速率等，但在处理函数的二阶近似时会面临计算量过大，以及实现过程复杂、兼容性差等缺陷。而PPO 算法具备 Policy Gradient、TRPO 的部分优点，采样数据和使用随机梯度上升方法优化代替目标函数之间交替进行，虽然标准的策略梯度方法对每个数据样本执行一次梯度更新，但 PPO 提出新目标函数，可以实现小批量更新。

  </details>
