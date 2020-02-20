
# TRPO and PPO -- A Reading Summary

## Introduction
Generally speaking, goal of reinforcement learning is to find an optimal behaviour strategy which maximizes rewards. Policy gradient methods are essential techniques in RL that directly optimize the parameterized policy by using an estimator of the gradient of the expected cost. In policy gradient methods, large step often leads to disasters, hence we should avoid  parameter updates that change the policy too much at one step. TRPO can improve training stability by constraining our step length to be within a “trust region”. However, since TRPO needs a KL divergence constraint on the size of update in each iteration, the implementation is complicated, hence another method, PPO, which simplifies TRPO a lot while emulating it, was proposed. Based on the fact that a certain surrogate objective forms a pessimistic bound on the performance of the policy, instead of using a KL divergence constraint, PPO uses a penalty.

## TRPO
**Keywords :** MM algorithm, monotonic improvement, 2nd-order optimization, conjugate gradient method, Fisher Informational Matrix.

### Introduction to TRPO
An essential idea of TRPO is monotinic improvement guarantee for policy, and to achieve monotonic improvement, an intuitive thought is to decompose the reward (or cost) of new policy into reward (or cost) of old policy plus an ''advantage term''.

Let \(\mathcal{S}\) be the finite set of states, \(\mathcal{A}\) be the finite set of actions, \(P:\mathcal{S}\times \mathcal{A}\times\mathcal{S}\rightarrow \mathbb{R}\) be the transition probability distribution, \(c:\mathcal{S}\rightarrow \mathbb{R}\) be the cost function, \(\rho_{0}:\mathcal{S}\rightarrow\mathbb{R}\) be the distribution of the initial state \(s_{0}\), \(\gamma\in (0,1)\) be the discount factor.
