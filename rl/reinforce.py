import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


def discounted_reward_to_go(
    rewards: torch.Tensor,
    dones: torch.Tensor | None = None,
    gamma: float = 0.99,
) -> torch.Tensor:
    """
    Compute discounted reward-to-go.

    Args:
        rewards: shape (T,) or (T, B)
        dones: shape (T,) or (T, B), True if episode terminates after this step.
        gamma: discount factor.

    Returns:
        returns: same shape as rewards.
    """
    rewards = rewards.float()
    original_was_1d = rewards.ndim == 1
    if original_was_1d:
        rewards = rewards[:, None]

    T, B = rewards.shape
    if dones is None:
        dones = torch.zeros_like(rewards, dtype=torch.bool)
    else:
        if dones.ndim == 1:
            dones = dones[:, None]
        dones = dones.bool()

    ##########################################################
    ## TODO: Compute the discounted return from each step
    returns = torch.zeros_like(rewards)  # (T, B)
    running_return = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)

    for t in reversed(range(T)):
        not_done = (~dones[t]).float()
        running_return = rewards[t] + gamma * running_return * not_done
        returns[t] = running_return
    ##########################################################

    if original_was_1d:
        returns = returns.squeeze(1)

    return returns


def reinforce_update_with_baseline(
    policy: nn.Module,
    value_fn: nn.Module,
    optimizer: optim.Optimizer,
    states: torch.Tensor,
    actions: torch.Tensor,
    returns: torch.Tensor,
    value_coef: float = 0.5,
) -> dict:
    states = states.float()
    actions = actions.long()
    returns = returns.float()

    logits = policy(states)
    dist = Categorical(logits=logits)
    log_probs = dist.log_prob(actions)

    values = value_fn(states).squeeze(-1)

    advantages = returns - values

    policy_loss = -(log_probs * advantages.detach()).mean()
    value_loss = ((values - returns) ** 2).mean()

    loss = policy_loss + value_coef * value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        "loss": loss.item(),
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "mean_return": returns.mean().item(),
    }


rewards = torch.tensor([1.0, 1.0, 1.0])
dones = torch.tensor([False, False, True])

assert torch.allclose(
    discounted_reward_to_go(rewards, dones, gamma=0.9), torch.tensor([2.71, 1.9, 1.0])
), "Discounted return wrong"

print("Passed!")
