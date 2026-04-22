import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal


# ==========================================
# EXERCISE 1: Generalized Advantage Estimation
# ==========================================
def compute_gae(rewards, values, dones, last_value, gamma=0.99, gae_lambda=0.95):
    """
    Computes Generalized Advantage Estimation (GAE) and returns.

    Args:
        rewards (Tensor): shape (T,)
        values (Tensor): shape (T,)
        dones (Tensor): shape (T,)
        last_value (Tensor): shape (1,)
        gamma (float): Discount factor
        gae_lambda (float): GAE lambda parameter

    Returns:
        advantages (Tensor): shape (T,)
        returns (Tensor): shape (T,)
    """
    # GAE formula: \sum_t (lambda * gamma)^t (r_t + gamma * V(s_t+1) - V(s_t))
    T = len(rewards)
    advantages = torch.zeros((T,))
    last_gae = 0
    for t in reversed(range(T)):
        next_value = values[t + 1] if t + 1 < T else last_value
        target = rewards[t] + gamma * (1 - dones[t]) * next_value - values[t]
        advantages[t] = last_gae = (
            target + (1 - dones[t]) * gae_lambda * gamma * last_gae
        )
    returns = advantages + values

    return advantages, returns


# ==========================================
# EXERCISE 2: Continuous Actor-Critic
# ==========================================
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # TODO: Define the critic network (scalar output)
        self.critic = nn.Linear(state_dim, 1)

        # TODO: Define the actor network (mean output)
        self.actor_mean = nn.Linear(state_dim, action_dim)

        # TODO: Define the learnable standard deviation (state-independent)
        # Hint: nn.Parameter
        self.actor_logstd = nn.Parameter(torch.ones(action_dim))

    def get_action_and_value(self, state, action=None):
        """
        Returns sampled action, log_prob, entropy, and value.
        If action is provided, evaluate that specific action instead of sampling.
        """
        # TODO: Get value from critic
        value = self.critic(state)

        # TODO: Get action distribution from actor
        # Hint: Extract mean, extract std (exp of logstd), create Normal distribution
        action_mean = self.actor_mean(state)
        action_dist = Normal(action_mean, torch.exp(self.actor_logstd))

        # TODO: Sample action (if not provided), compute log_prob and entropy
        # VERY IMPORTANT: log_prob should be summed over the action dimensions
        if action is None:
            action = action_dist.sample()
        log_prob = action_dist.log_prob(action).sum(1)
        entropy = action_dist.entropy().sum(1)

        return action, log_prob, entropy, value


# ==========================================
# EXERCISE 3: PPO Surrogate Loss
# ==========================================
def ppo_loss(
    log_probs,
    old_log_probs,
    advantages,
    returns,
    values,
    clip_coef=0.2,
    vf_coef=0.5,
    ent_coef=0.01,
    entropy=None,
):
    """
    Computes the PPO loss.
    """
    ratio = torch.exp(log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = ratio.clip(1 - clip_coef, 1 + clip_coef) * advantages
    policy_loss = -torch.minimum(surr1, surr2).mean()
    value_loss = ((returns - values) ** 2).mean()
    if entropy is None:
        entropy_loss = 0
    else:
        entropy_loss = -ent_coef * entropy.mean()
    total_loss = policy_loss + value_loss + entropy_loss

    return total_loss, policy_loss, value_loss


# ==========================================
# EXERCISE 4: Minibatch Training Loop
# ==========================================
def ppo_update(
    model,
    optimizer,
    states,
    actions,
    old_log_probs,
    advantages,
    returns,
    batch_size=32,
    epochs=4,
):
    """
    Executes the PPO optimization loop.
    """
    dataset_size = states.size(0)

    for epoch in range(epochs):
        # TODO: Generate randomized indices for minibatches
        indices = torch.randperm(dataset_size)

        for start_idx in range(0, dataset_size, batch_size):
            # TODO: Extract minibatch indices
            mb_indices = indices[start_idx : start_idx + batch_size]

            # TODO: Extract minibatch data
            mb_states = states[mb_indices]
            mb_actions = actions[mb_indices]
            mb_old_log_probs = old_log_probs[mb_indices]
            mb_advantages = advantages[mb_indices]
            mb_returns = returns[mb_indices]

            # TODO: Forward pass to get new log_probs, entropy, and values
            _, mb_log_probs, mb_entropy, mb_values = model.get_action_and_value(
                mb_states, mb_actions
            )

            # TODO: Compute loss using ppo_loss
            # Potential gotchas: shape of NN output likely (B, 1) and shape of returns likely (B,)
            total_loss, _, _ = ppo_loss(
                mb_log_probs,
                mb_old_log_probs,
                mb_advantages,
                mb_returns,
                mb_values,
                entropy=mb_entropy,
            )

            # TODO: Optimizer step (zero_grad, backward, step)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()


# ==========================================
# COMPREHENSIVE TEST SUITE
# ==========================================
def run_tests():
    print("Running PPO Implementation Tests...\n")

    # --- Test 1: GAE ---
    try:
        rewards = torch.tensor([1.0, 2.0, -1.0])
        values = torch.tensor([0.5, 1.5, -0.5])
        dones = torch.tensor([0.0, 0.0, 1.0])
        last_val = torch.tensor([0.0])
        gamma, gae_lambda = 0.9, 0.8

        adv, ret = compute_gae(rewards, values, dones, last_val, gamma, gae_lambda)

        # Mathematical derivation for checks:
        # delta_2 = -1.0 + 0.9 * 0.0 * 0 - (-0.5) = -0.5
        # delta_1 = 2.0 + 0.9 * (-0.5) * 1 - 1.5 = 0.05
        # delta_0 = 1.0 + 0.9 * 1.5 * 1 - 0.5 = 1.85
        # A_2 = -0.5
        # A_1 = 0.05 + 0.9 * 0.8 * (-0.5) = -0.31
        # A_0 = 1.85 + 0.9 * 0.8 * (-0.31) = 1.6268
        expected_adv = torch.tensor([1.6268, -0.31, -0.5])
        expected_ret = expected_adv + values

        assert torch.allclose(adv, expected_adv, atol=1e-4), (
            f"GAE Advantages incorrect.\nExpected: {expected_adv}\nGot: {adv}"
        )
        assert torch.allclose(ret, expected_ret, atol=1e-4), (
            "Returns must exactly equal Advantages + Values."
        )
        print("✅ Exercise 1 (GAE) Passed!")
    except Exception as e:
        print(f"❌ Exercise 1 Failed: {e}")

    # --- Test 2: Actor-Critic ---
    try:
        state_dim, action_dim = 4, 2
        model = ActorCritic(state_dim, action_dim)
        dummy_state = torch.randn(10, state_dim)  # Batch of 10

        action, log_prob, entropy, value = model.get_action_and_value(dummy_state)

        assert action.shape == (10, action_dim), (
            f"Action shape should be (10, 2), got {action.shape}"
        )
        # CRITICAL CHECK: In continuous control, independent action dims must be summed for joint log_prob
        assert log_prob.shape == (10,), (
            f"Log prob shape should be (10,). Did you sum over the action dimension? Got {log_prob.shape}"
        )
        assert entropy.shape == (10,), (
            f"Entropy shape should be (10,). Did you sum over action dim? Got {entropy.shape}"
        )
        assert value.shape == (10, 1), (
            f"Value shape should be (10, 1), got {value.shape}"
        )

        _, log_prob_eval, _, _ = model.get_action_and_value(dummy_state, action)
        assert torch.allclose(log_prob, log_prob_eval), (
            "Evaluating a given action should perfectly match the sampled log_prob."
        )
        print("✅ Exercise 2 (Actor-Critic) Passed!")
    except Exception as e:
        print(f"❌ Exercise 2 Failed: {e}")

    # --- Test 3: PPO Loss ---
    try:
        # Construct exact scenarios to test clipping logic
        old_log_probs = torch.tensor([0.0, 0.0, 0.0])
        # Ratios will be: [1.5, 0.5, 1.5]
        log_probs = torch.tensor(
            [
                torch.log(torch.tensor(1.5)),
                torch.log(torch.tensor(0.5)),
                torch.log(torch.tensor(1.5)),
            ]
        )
        # Advantages: Good action, Bad action, Bad action
        advantages = torch.tensor([1.0, -1.0, -1.0])
        returns = torch.tensor([1.0, -1.0, -1.0])
        values = torch.tensor([0.5, -0.5, -0.5])
        entropy = torch.tensor(1.0)

        total_loss, p_loss, v_loss = ppo_loss(
            log_probs,
            old_log_probs,
            advantages,
            returns,
            values,
            clip_coef=0.2,
            vf_coef=0.5,
            ent_coef=0.01,
            entropy=entropy,
        )

        # Policy Loss Verification:
        # Index 0: ratio=1.5, Adv=1.0. Clipped to 1.2 * 1.0 = 1.2
        # Index 1: ratio=0.5, Adv=-1.0. Clipped to 0.8 * -1.0 = -0.8
        # Index 2: ratio=1.5, Adv=-1.0. Unclipped (-1.5) < Clipped (-1.2), so uses Unclipped = -1.5
        # Expected mean policy loss (negated) = -mean(1.2, -0.8, -1.5) = -mean(-1.1) = 0.3666...
        expected_p_loss = torch.tensor(0.3666666)
        assert torch.allclose(p_loss, expected_p_loss, atol=1e-4), (
            f"Policy clipping logic incorrect. Expected {expected_p_loss}, got {p_loss}"
        )

        # VF Loss Verification:
        # MSE between values [0.5, -0.5, -0.5] and returns [1.0, -1.0, -1.0] = mean(0.25, 0.25, 0.25) = 0.25
        expected_v_loss = torch.tensor(0.25)
        assert torch.allclose(v_loss, expected_v_loss, atol=1e-4), (
            "Value loss MSE calculation incorrect."
        )

        print("✅ Exercise 3 (PPO Loss) Passed!")
    except Exception as e:
        print(f"❌ Exercise 3 Failed: {e}")

    # --- Test 4: Minibatch Update ---
    try:
        state_dim, action_dim = 4, 2
        model = ActorCritic(state_dim, action_dim)
        # Mocking the architecture to allow forward passes
        model.critic = nn.Sequential(nn.Linear(state_dim, 16), nn.Linear(16, 1))
        model.actor_mean = nn.Sequential(
            nn.Linear(state_dim, 16), nn.Linear(16, action_dim)
        )
        model.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # Create dataset
        N = 64
        states = torch.randn(N, state_dim)
        actions = torch.randn(N, action_dim)
        old_log_probs = torch.randn(N)
        advantages = torch.randn(N)
        returns = torch.randn(N)

        # Save initial weights to verify gradients flowed
        initial_weights = copy.deepcopy(model.actor_mean[0].weight.data)

        ppo_update(
            model,
            optimizer,
            states,
            actions,
            old_log_probs,
            advantages,
            returns,
            batch_size=16,
            epochs=2,
        )

        new_weights = model.actor_mean[0].weight.data
        assert not torch.allclose(initial_weights, new_weights), (
            "Model parameters did not change. Backward pass or optimizer step missing."
        )
        print("✅ Exercise 4 (Minibatch Loop) Passed!")
    except Exception as e:
        print(f"❌ Exercise 4 Failed: {e}")


if __name__ == "__main__":
    run_tests()
