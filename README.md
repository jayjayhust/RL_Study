# RL学习（基于GYM）

## 环境准备 
- 安装
```
pip install gym
```

## 参考教程
- 视频教程
  - [Gymnasium (Deep) Reinforcement Learning](https://www.youtube.com/playlist?list=PL58zEckBH8fCt_lYkmayZoR9XfDCW9hte)[代码](https://github.com/johnnycode8/gym_solutions)
  - 很好的图文解释RL视频系列：[Reinforcement Learning from scratch](https://www.youtube.com/watch?v=vXtfdGphr3c)

## 概念学习
- 基本概念
  - <details><summary><strong>Q-Learning</strong></summary>

  Q-learning是一种经典的强化学习算法，用于解决马尔可夫决策过程（Markov Decision Process，MDP）中的控制问题。它是基于值迭代（Value Iteration）的思想，通过估计每个状态动作对的价值函数Q值来指导智能体在每个状态下选择最佳的动作。
  其算法的基本思想跟主要优势如下：Q-Learning是强化学习算法中value-based的算法，Q即为Q（s，a），就是在某一个时刻的state状态下，采取动作a能够获得收益的期望，环境会根据agent的动作反馈相应的reward奖赏，所以算法的主要思想就是将state和action构建成一张Q_table表来存储Q值，然后根据Q值来选取能够获得最大收益的动作。
  Q-learning的主要优势就是使用了时间差分法（融合了蒙特卡洛和动态规划）能够进行off-policy的学习，使用贝尔曼方程可以对马尔科夫过程求解最优策略.
  
  </details>