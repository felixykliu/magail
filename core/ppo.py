import torch
from torch.autograd import Variable


def ppo_step(policy_net, value_net, optimizer_policy, optimizer_value, optim_value_iternum, states, actions,
             returns, advantages, fixed_log_probs, lr_mult, lr, clip_epsilon, l2_reg, p_i = None, i_iter = 10000):

    optimizer_policy.lr = lr * lr_mult
    optimizer_value.lr = lr * lr_mult
    clip_epsilon = clip_epsilon * lr_mult

    """update critic"""
    values_target = Variable(returns)
    for i in range(optim_value_iternum):
        values_pred, _ = value_net(Variable(states), test = False)
        value_loss = (values_pred - values_target).pow(2).mean()
        if p_i == 0 and i == 0:
            print(value_loss)
        '''
        if p_i == 0 and i == 0:
            print(states[:, 0, :])
            print(hidden_layers.cpu().data[:, 0])
            print(values_pred.cpu().squeeze().data[:, 0])
            print(value_loss)
            print(value_net.parameters())
        '''
        # weight decay
        for param in value_net.parameters():
            value_loss += param.pow(2).sum() * l2_reg
        optimizer_value.zero_grad()
        value_loss.backward()
        optimizer_value.step()

    """update policy"""
    if i_iter > 5:
        advantages_var = Variable(advantages)
        log_probs = policy_net.get_log_prob(Variable(states), Variable(actions))
        ratio = torch.exp(log_probs - Variable(fixed_log_probs))
        surr1 = ratio * advantages_var
        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages_var
        policy_surr = -torch.min(surr1, surr2).mean()
        optimizer_policy.zero_grad()
        policy_surr.backward()
        torch.nn.utils.clip_grad_norm(policy_net.parameters(), 40)
        '''
        if p_i == 0:
            ## when clip = 0.1, grads are ~10-100 times smaller than values
            for par in policy_net.parameters():
                print("var: ", par)
                print("grad: ", par.grad)
            exit()
        '''
        optimizer_policy.step()
