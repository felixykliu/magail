

def estimate_advantages(rewards, values, gamma, tau, use_gpu):
    if use_gpu:
        rewards, values = rewards.cpu(), values.cpu()
    tensor_type = type(rewards)
    returns = tensor_type(rewards.size(0), rewards.size(1), 1)
    deltas = tensor_type(rewards.size(0), rewards.size(1), 1)
    advantages = tensor_type(rewards.size(0), rewards.size(1), 1)

    prev_return = 0
    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + gamma * prev_return
        deltas[i] = rewards[i] + gamma * prev_value - values[i]
        advantages[i] = deltas[i] + gamma * tau * prev_advantage
        
        prev_return = returns[i]
        prev_value = values[i]
        prev_advantage = advantages[i]

    advantages = (advantages - advantages.mean()) / advantages.std()

    if use_gpu:
        advantages, returns = advantages.cuda(), returns.cuda()
    return advantages, returns
