## Questions

1. You monitor the approximate KL divergence between $\pi_{\theta}$ and $\pi_{\theta_{old}}$ during the inner optimization epochs. The KL divergence consistently spikes above 0.05 (or your target threshold), and the agent's performance collapses. What are the two primary parameters you should decrease to stabilize this?

2. Your agent's policy entropy drops to near zero very early in training, well before it has adequately explored the environment. It locks into a highly suboptimal, deterministic policy. Which parameter must be increased?

3. You are using a shared neural network trunk for both the actor and the critic. The initial value loss is extremely high, and the resulting gradients flowing back through the shared layers completely destruct the actor's early representations. Which parameter should you decrease to isolate this issue?

4. Your environment features highly stochastic, noisy rewards. The advantage estimates show massive variance, destabilizing the gradient updates. To reduce this variance (at the cost of introducing more bias into the advantage estimation), in which direction should you adjust the GAE $\lambda$ parameter?

5. You notice that by the 8th epoch of your inner training loop, the policy is over-fitting to the current batch of rollout data, and performance on the next rollout drops. Aside from simply hard-coding fewer epochs, what dynamic early-stopping criterion is conventionally used in modern PPO implementations to prevent this?

6. You increase the environment rollout length (the horizon, $T$) to capture longer-term dependencies. Suddenly, your optimization steps become highly unstable. Assuming the learning rate remains unchanged, what parameter relating to the optimization phase must be scaled up to maintain stable, accurate gradient estimates?

7. You are training a continuous control agent and decide to normalize your advantages to have a mean of 0 and a standard deviation of 1. However, you accidentally apply this normalization at the minibatch level (size 64) rather than across the entire rollout buffer (size 2048) before the inner optimization epochs. What is the most likely pathological symptom you will observe?

8. Your environment yields small step penalties ($-1$) and a massive success reward ($+10,000$). During early training, when the agent accidentally hits the goal, the critic's MSE loss explodes to infinity. In the very next epoch, the actor policy completely collapses, outputting NaNs. You are using standard PPO with a shared network trunk. What is the most principled architectural or algorithmic fix?

9. Your continuous control PPO agent controls a drone throttle bounded strictly between $[0, 1]$. Your actor network outputs a Gaussian mean, and the action is simply clipped by numpy.clip(action, 0, 1) before being sent to the environment. If the network initializes with a mean of $0$ and a standard deviation of $1.0$, what hidden mathematical issue will plague early training?

10. You implement a target KL divergence early stopping mechanism to prevent over-optimization during the inner loop. You mistakenly set the target KL to a very strict threshold of $0.001$ instead of the conventional $0.01$ to $0.05$. What training dynamic are you most likely to observe?

11. Your robotic locomotion task has an artificial time limit of 1000 steps to prevent infinite episodes. You correctly flag done=True when the robot physically falls over, but you ALSO flag done=True exactly at step 1000. In your GAE implementation, you hardcode a bootstrapped value of $0$ for all done=True states. How will this specific implementation detail affect the agent's behavior as it approaches step 1000?

## Answer Key (Gemini generated)

1. learning rate and clipping coefficient
2. increase entropy coefficient
3. value loss coefficient
4. decrease GAE
5. KL divergence monitoring and early stopping if KL divergence is above a pre-set threshold.
6. increase the minibatch size
7. Minibatch advantages will have artificially inflated variance, distorting the true relative quality of actions and causing the policy updates to oscillate wildly.
8. Apply reward scaling by dividing rewards by a running standard deviation, ensuring the value targets stay within a reasonable numerical range.
9. Boundary effect: 50% of sampled actions will be negative, clipped to 0 by the environment, but the policy will update as if it took the exact negative continuous values, creating a severe disconnect. The fix is to apply a tanh to the sampled action instead of a hard clipping. The likelihood can be found via a change of variables.
10. The KL bound will likely be hit after the very first minibatch update, causing the PPO algorithm to fall back to a naive REINFORCE update.
11. The robot would think that there is no need to e.g. maintain upright posture at step 1000. As a result, it can start to myopically optimize for short-term rewards in the final steps. The fix is to differentiate `done` from `truncated`. When computing the GAE estimate, the next step value will be $0$ if `done` and $V(s_T)$ if `truncated`.