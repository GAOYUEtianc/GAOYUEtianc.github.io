
---
title: "2020 Resolutions"
date: 2020-02-21
draft: false
type: "post"
hidden: false
mathjax: true
markup: mmark

---



# TRPO and PPO -- A Reading Summary

## Introduction
Generally speaking, goal of reinforcement learning is to find an optimal behaviour strategy which maximizes rewards. Policy gradient methods are essential techniques in RL that directly optimize the parameterized policy by using an estimator of the gradient of the expected cost. In policy gradient methods, large step often leads to disasters, hence we should avoid  parameter updates that change the policy too much at one step. TRPO can improve training stability by constraining our step length to be within a “trust region”. However, since TRPO needs a KL divergence constraint on the size of update in each iteration, the implementation is complicated, hence another method, PPO, which simplifies TRPO a lot while emulating it, was proposed. Based on the fact that a certain surrogate objective forms a pessimistic bound on the performance of the policy, instead of using a KL divergence constraint, PPO uses a penalty.

## TRPO
**Keywords :** MM algorithm, monotonic improvement, 2nd-order optimization, conjugate gradient method, Fisher Informational Matrix.

### Introduction to TRPO
An essential idea of TRPO is monotinic improvement guarantee for policy, and to achieve monotonic improvement, an intuitive thought is to decompose the reward (or cost) of new policy into reward (or cost) of old policy plus an ''advantage term''.

Let $\mathcal{S}$ be the finite set of states, $\mathcal{A}$ be the finite set of actions, $P:\mathcal{S}\times \mathcal{A}\times\mathcal{S}\rightarrow \mathbb{R}$ be the transition probability distribution, $c:\mathcal{S}\rightarrow \mathbb{R}$ be the cost function, $\rho_{0}:\mathcal{S}\rightarrow\mathbb{R}$ be the distribution of the initial state $s_{0}$, $\gamma\in (0,1)$ be the discount factor. 

Here are some standard definitions of the state-action value function $Q_{\pi}$, the value function $V_{\pi}$, and the advantage function $A_{\pi}$ :
$$\begin{array}{cc}
Q_{\pi} (s_{t}, a_{t}) &=& \mathbb{E}_{s_{t+1}, a_{t+1}, ...}\left[\sum_{l=0}^{\infty}\gamma^{l}c(s_{t+l})\right]\\
V_{\pi}(s_{t}) &=& \mathbb{E}_{a_{t},s_{t+1},...}\left[\sum_{l=0}^{\infty}\gamma^{l}c(s_{t+l})\right]\\
A_{\pi}(s,a) &=& Q_{\pi}(s,a) - V_{\pi}(s), \;where\; a_{t}\sim \pi(a_{t}, s_{t}), s_{t+1}\sim P[s_{t+1}|s_{t},a_{t}]
\end{array}$$

Kakade \& Langford raised up an identity that express the expected cost of another policy $\tilde{\pi}$ in terms of the accumulated advantage over $\pi$ :
$$\begin{array}{cc}
\eta(\tilde{\pi}) &=& \eta(\pi) + \mathbb{E}_{s_{0},a_{0},s_{1},a_{1},...}\left[\sum_{t=0}^{\infty}\gamma^{t}A_{\pi}(s_{t},a_{t})\right]
\end{array}$$
where $s_{0}\sim \rho_{0}(s_{0})$, $a_{t}\sim \tilde{\pi}(a_{t}|s_{t})$, $s_{t+1}\sim P[s_{t+1}|s_{t},a_{t}]$.

The proof is as follows (let $\tau|\tilde{\pi}$ denote the trajectory $\tau = (s_{0},a_{0},s_{1},a_{1})$ generated from $\tilde{\pi}$):

$$\begin{array}{c}
& \mathbb{E}_{\tau|\tilde{\pi}}\left[\sum_{t=0}^{\infty}\gamma^{t}A_{\pi}(s_{t},a_{t})\right] \\
=&\mathbb{E}_{\tau|\tilde{\pi}}\left[\sum_{t=0}^{\infty}\gamma^{t} (r(s_{t}) +\gamma V_{\pi}(s_{t+1})-V_{\pi}(s_{t}))\right]\\
=&\mathbb{E}_{\tau|\tilde{\pi}}\left[-V_{\pi}(s_{0})+\sum_{t=0}^{\infty}\gamma^{t}\cdot r(s_{t})\right]\\
=& -\mathbb{E}_{s_{0}}\left[V_{\pi}(s_{0})\right] +\mathbb{E}_{\tau|\tilde{\pi}}\left[\sum_{t=0}^{\infty}\gamma^{t}\cdot r(s_{t})\right] \\
=& -\eta(\pi) + \eta(\tilde{\pi})
\end{array}$$

Let $\rho_{\pi}(s)=P[s_{0}=s]+\gamma P[s_{1}=s]+\gamma^{2} P[s_{2}=s]+...$ be the (unnormalized) discounted visitaion frequencies. Then,

$$\begin{array}{cc}
\eta(\tilde{\pi}) &=& \eta(\pi) +\mathbb{E}_{\tau|\tilde{\pi}}\left[\sum_{t=0}^{\infty}\gamma^{t}A_{\pi}(s_{t},a_{t})\right]\\
&=& \eta(\pi) + \sum_{t=0}^{\infty}\sum_{s}P[s_{t}=s|\tilde{\pi}]\sum_{a}\tilde{\pi}(a|s)\gamma^{t}A_{\pi}(s,a)\\
&=& \eta(\pi) + \sum_{s}\sum_{t=0}^{\infty}\gamma^{t}P[s_{t}=s|\tilde{\pi}]\sum_{a}\tilde{\pi}(a|s)A_{\pi}(s,a)\\
&=& \eta(\pi) + \sum_{s}\rho_{\tilde{\pi}}(s)\sum_{a}\tilde{\pi}(a|s)A_{\pi}(s,a) 
\end{array}$$

However, since we don't have $\rho_{\tilde{\pi}}$, eqn (13) is difficult to optimize, hence, we introduce a local approximation to $\eta$, and note that this is the first 'approximation trick' in TRPO :
$$\begin{array}{cc}
L_{\pi}(\tilde{\pi}) = \eta(\pi) + \sum_{s}\rho_{\pi}(s)\sum_{a}\tilde{\pi}(a|s)A_{\pi}(s,a)
\end{array}$$

Compared to the equations above, $L_{\pi}$ uses $\rho_{\pi}$ as a local approximation of $\rho_{\tilde{\pi}}$, intuitively this is reasonable since within 1 iteration, the new policy won't be that different from the old one, so the visitation frequencies under those two policies would be similar. More precisely, if the parameterized policy $\pi_{\theta}(a|s)$ is a differentiable function of the parameter vector $\theta$, then $L_{\pi}$ matches $\eta$ to the first order, i.e., for any $\theta_{0}$,

$$\begin{array}{cc}
L_{\pi_{\theta_{0}}}(\pi_{\theta_{0}}) &=& \eta(\pi_{\theta_{0}})\\
\nabla_{\theta}L_{\pi_{\theta_{0}}}(\pi_{\theta})|_{\theta=\theta_{0}} &=& \nabla_{\theta}\eta(\pi_{\theta})|_{\theta=\theta_{0}}
\end{array}$$

According to the thoughts behind Minorize-Maximization algorithms, by iteratively maximizing lower bound function of the expected reward, an algorithm can guarantee that any policy update always improve the expected reward. Let 
$$\begin{array}{cc}
\pi^{'}&=&\underset{\pi^{'}}{\argmax}\;L_{\pi_{old}}(\pi^{'})\\
\pi_{new}(a|s) &=& (1-\alpha)\pi_{old}(a|s)+\alpha\pi^{'}(a|s)
\end{array}$$
Kakade \& Langford derived the following lower bound :$$\begin{array}{cc}
\eta(\pi_{new}) &\geq & L_{\pi_{old}}(\pi_{new}) - \frac{2\epsilon\gamma}{(1-\gamma)^{2}}\alpha^{2}\\
where\;\epsilon &=& \underset{s}{\max}\;|\mathbb{E}_{a\sim\pi^{'}(a|s)}[A_{\pi}(s,a)]|
\end{array}$$
However, mixer policies are rarely used in practice, so based on this bound, Schulman et al. extended this to general stochastic policies by replacing $\alpha$ in eqn (19) by $ D_{TV}^{max}(\pi_{old}, \pi_{new})$, where $D_{TV}^{max}(\pi,\tilde{\pi})=\underset{s}{\max}\;D_{TV}(\pi(\cdot|s)||\tilde{\pi}(\cdot|s))$ and $D_{TV}(p||q)=\frac{1}{2}\sum_{i}|p_{i}-q_{i}|$ be the total variation divergence. Pollard has proved that $D_{KL}^{max}(\pi,\tilde{\pi})=\underset{s}{\max}\;D_{KL}(\pi(\cdot|s)||\tilde{\pi}(\cdot|s))$. Hence in the lower bound we can replace $D_{TV}^{max}(\pi_{old}, \pi_{new})$ by $D_{KL}^{max}(\pi_{old}, \pi_{new})$.

An intuitive thought is that by iteratively returning the policies maximizing $L_{\pi_{old}}(\pi)-\frac{2\epsilon\gamma}{(1-\gamma)^{2}}\;D_{KL}^{max}(\pi_{old}, \pi)$, we can guarantee monotonic improvement. However, computing the maximum KL divergence needs to iterate over all states, which is super intractable, fortunately, Schulman et al. showed that here mean KL divergence over state space is a valid approximation to maximum KL divergence <font color=red>empirically. (I have a doubt here, I didn't find any theoretical evidence that replacing max KL divergence by average KL divergence is 'theoretically valid'. The paper only mentioned that under those two types of divergence, the algorithms have similar performance empirically.)</font>

Finally, we arrive at the optimization problem :$$\begin{array}{c}
\underset{\theta}{maximize}\left[L_{\theta_{old}}(\theta) - \frac{2\epsilon\gamma}{(1-\gamma)^{2}}\cdot\bar{D}_{KL}^{\rho_{old}}(\theta_{old}, \theta)\right]
\end{array}$$
where  $\bar{D}_{KL}^{\rho}(\theta_{1},\theta_{2})=\mathbb{E}_{s\sim\rho}\left[D_{KL}(\pi_{\theta_{1}}(\cdot|s)||\pi_{\theta_{2}}(\cdot|s))\right]$.

There are two basic iterative approaches to find a local minimum/maximum of an objective function, line search and trust region. The method adopted in this paper is trust region, i.e.,determine the maximum step size we want to explore then locate the optimal point within this trust region :$$\begin{array}{cc}\underset{\theta}{maximize}\; \sum_{s}\rho_{\theta_{old}}(s)\sum_{a}\pi_{\theta}(a|s)A_{\theta_{old}}(s,a)\\
 subject\;to\; \bar{D}_{KL}^{\rho_{\theta_{old}}}(\theta_{old},\theta) \leq \delta
\end{array}$$

Consider the case when we are doing off-policy RL, the policy $q$ used for collecting trajectories is different from the policy to optimize. The mismatch between the training data distribution and the true policy state distribution is compensated by importance sampling estimator. Note that since true rewards are usually unknown, we use an estimated advantage $\hat{A}(\cdot)$ (by performing a rollout) instead of $A(\cdot)$ :$$\begin{array}{c}
 & \sum_{s}\rho_{\theta_{old}}(s)\sum_{a}\pi_{\theta}(a|s)\hat{A}_{\theta_{old}}(s,a) \\
 =& \sum_{s}\rho_{\theta_{old}}(s)\sum_{a}q(a|s)\frac{\pi_{\theta}(a|s)}{q(a|s)}\hat{A}_{\theta_{old}}(s,a)\\
 =& \mathbb{E}_{s\sim \rho_{old},a\sim q}\left[\frac{\pi_{\theta}(a|s)}{q(a|s)}\hat{A}_{\theta_{old}}(s,a)\right]
\end{array}$$

In continuous control problems, it's better to use $\pi_{\theta_{old}}$ as behaviour policy. 

After getting the objective function, we can solve this optimization problem with a 2nd order approximation of the KL divergence, 1st order approximation of loss and natural gradient descent. Note that the natural policy gradient descent requires computing of Fisher Information Matrix and its inverse, which is expensive. 

## PPO

A major disadvantage of TRPO is that it's computationally expensive, Schulman et al. proposed proximal policy optimization (PPO) to simplify TRPO by using a clipped surrogate objective while retaining similar performance. Compared to TRPO, PPO is simpler, faster, and more sample efficient. 

Let $r_{t}(\theta) = \frac{\pi_{\theta}(a_{t}|s_{t})}{\pi_{\theta_{old}}(a_{t}|s_{t})}$, then the surrogate objective of TRPO is $$\begin{array}{cc}
L^{CPI}(\theta) = \hat{\mathbb{E}}_{t}\left[r_{t}(\theta)\hat{A}_{t}\right]
\end{array}$$
Note that without a limitation on the distance between $\theta$ and $\theta_{old}$, maximizing $L^{CPI}$ will lead to excessively large parameter updates and big policy ratios. PPO penalize changes to policy by forcing $r_{t}(\theta)$ to stay within interval $[1-\epsilon, 1+\epsilon]$ (where $\epsilon$ is a hyperparameter). The objective function is : $$\begin{array}{cc}
L^{CLIP}(\theta) = \hat{\mathbb{E}}_{t}\left[\min\; (r_{t}(\theta)\hat{A}_{t},\; clip(r_{t}(\theta),1-\epsilon, 1+\epsilon)\hat{A}_{t})\right]
\end{array}$$
where the function $clip(r_{t}(\theta),1-\epsilon, 1+\epsilon)$ clips the ratio $r_{t}(\theta)$ within $[1-\epsilon, 1+\epsilon]$. The objective function takes the minimum between the clipped and unclipped objective, so the final objective is a pessimistic bound on the unclipped one. Gradient Descent like Adam can be used to optimize it.

When applying PPO on the network architecture with shared parameters for both policy and value function, the objective function can be further augmented with an error term on the value estimation and an entropy bonus to ensure sufficient exploration : $$\begin{array}{cc}
L_{t}^{LIP+VF+S}(\theta) = \hat{\mathbb{E}}_{t}\left[L_{t}^{CLIP}(\theta) - c_{1}(V_{\theta}(s_{t}) - V_{t}^{target})^{2} + c_{2}S[\pi_{\theta}](s_{t})\right]
\end{array}$$
where $c_{1},c_{2}$ are coefficients, and $S$ is the entropy bonus. 

### Controversies in PPO

PPO is controversial though, it has shown great practical promise, however, there are works raising doubts about PPO. A work called 'Implementation Matters In Deep Policy Gradients : A Case Study On PPO And TRPO' investigated the consequences of 'code-level optimizations' in detail. Concretely, PPO's code-optimizations are significantly more important in terms of final reward, instead of the choice of general training algorithm (TRPO vs. PPO), contradicting the belief that 'clipping tech' is the key innovation of PPO. Also, PPO enforces trust region by code-level optimizations instead of the clipping technique. Moreover, the clipping technique may not be necessary, PPO-NoCLIP algorithm, which uses code-level optimizations but no clipping mechanism, achieves similar results to PPO in terms of benchmark performance. 

## Reflections

This issue in PPO points to a broader problem: we don't really understand how the parts comprising deep RL algorithms impact agent training, either as individuals or as a whole. When designing deep RL algorithms, it's necessary for us to understand precisely how each component impacts agent training and designing algorithms in a modular manner. \\

We've also noticed that although in PPO the clipping technique is designed to limit the distance between $\theta$ and $\theta_{old}$, actually clipping technique does not enforce the KL region as it's supposed to. Then an overarching question is : To what degree does current practice in deep RL reflect the principles informing its development? Motivated by this question, the work 'A closer look at deep policy gradients' proposed a fine-grained analysis of state-of-the-art methods, and (sadly), the results show that the  behaviour of deep policy gradient algorithms often deviates from the prediction of their motivating framework : 
    1. Deep policy gradient methods operate with relatively poor estimates of the gradient, especially as task complexity increases and as training progresses. Better gradient estimates can require lower learning rates and can induce degenerate agent behaviour. 
    2. As training progresses, the surrogate objective becomes much less predictive of the true reward in the relevant sample regime, the underlying optimization landscape can be misleading.
    3. Learned value estimators does not accurately model the true value function, and the value networds reduce gradient estimation variance to a significantly smaller extent than the true value.

In conclusion, we need a more fine-grained understanding of deep RL algorithms, and to close the gap between the theory inspiring algorithms and the actual mechanisms, we need to either develop methods intimately bound up with theory, or build theory that can capture what makes existing policy gradient methods successful.

<font color=blue>Thanks Alex Lewandowski for recommending those brilliant papers!</font>

## References

John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz. Trust region policy optimization. In International Conference on Machine Learning (ICML), pages 1889–1897, 2015.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Logan Engstrom, Andrew Ilyas, Shibani Santurkar, Dimitris Tsipras, Firdaus Janoos, Larry Rudolph, Aleksander Madry. Implementation Matters in Deep RL: A Case Study on PPO and TRPO. https://openreview.net/forum?id=r1etN1rtPB.

Andrew Ilyas, Logan Engstrom, Shibani Santurkar, Dimitris Tsipras, Firdaus Janoos, Larry Rudolph, Aleksander Madry. A Closer Look at Deep Policy Gradients. https://openreview.net/forum?id=ryxdEkHtPS