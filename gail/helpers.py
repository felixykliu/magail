from utils import *
from core.common import estimate_advantages
from torch.autograd import Variable
from torch import nn
import torch
from core.ppo import ppo_step
from core.trpo import trpo_step
import numpy as np
import os
import struct
from bball_data.utils import unnormalize, plot_sequences, plot_animation, plot_macrogoals, plot_heatmap
from core.common import estimate_advantages
import pickle

def update_dis_and_critic(discrim_net, optimizer_discrim, discrim_criterion, value_net, optimizer_value, exp_states, exp_actions, states, actions, returns, l2_reg, i_iter, dis_times, critic_times, use_gpu, update_discrim = True):
    if use_gpu:
        states, actions, returns, exp_states,  exp_actions = \
        states.cuda(), actions.cuda(), returns.cuda(), exp_states.cuda(), exp_actions.cuda()

    """update discriminator"""
    g_o_ave = 0.0
    e_o_ave = 0.0
    for _ in range(int(dis_times)):
        g_o, _ = discrim_net(Variable(states), Variable(actions))
        e_o, _ = discrim_net(Variable(exp_states), Variable(exp_actions))
        
        g_o_ave += g_o.cpu().data.mean()
        e_o_ave += e_o.cpu().data.mean()
        
        if update_discrim:
            optimizer_discrim.zero_grad()
            discrim_loss = discrim_criterion(g_o, Variable(zeros((states.shape[0], states.shape[1], 1)))) + \
                discrim_criterion(e_o, Variable(ones((exp_states.shape[0], exp_states.shape[1], 1))))
            discrim_loss.backward()
            optimizer_discrim.step()
    
    '''update value function'''
    values_target = Variable(returns)
    for i in range(int(critic_times)):
        values_pred, _ = value_net(Variable(states), test = False)
        value_loss = (values_pred - values_target).pow(2).mean()
        if i == 0:
            print("Value function loss: {:.4f}".format(value_loss.data.cpu().numpy()[0]))
        
        for param in value_net.parameters():
            value_loss += param.pow(2).sum() * l2_reg
        optimizer_value.zero_grad()
        value_loss.backward()
        optimizer_value.step()
    
    if dis_times > 0:
        return g_o_ave / dis_times, e_o_ave / dis_times, value_loss.data.cpu().numpy()[0]

def update_policy_ppo(policy_net, optimizer_policy, states, actions, advantages, fixed_log_probs, i_iter, clip_epsilon, use_gpu):
    if use_gpu:
        states, actions, advantages, fixed_log_probs = \
        states.cuda(), actions.cuda(), advantages.cuda(), fixed_log_probs.cuda()
    
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
    ## when clip = 0.1, grads are ~10-100 times smaller than values
    for par in policy_net.parameters():
        print("var: ", par)
        print("grad: ", par.grad)
    exit()
    '''
    optimizer_policy.step()

# pretrain policy
def pre_train_policy(policy_net, optimizer_policy, expert_states, expert_actions, size, train=True):
    # expert
    exp_ind = np.random.choice(expert_states.shape[0], size)
    sample_expert_states = torch.from_numpy(expert_states[exp_ind].copy()).transpose(0, 1)   ## size * seq_len * 22
    sample_expert_actions = torch.from_numpy(expert_actions[exp_ind].copy()).transpose(0, 1)

    if use_gpu:
        sample_expert_states, sample_expert_actions = sample_expert_states.cuda(), sample_expert_actions.cuda()
    
    fixed_log_probs = policy_net.get_log_prob(Variable(sample_expert_states), Variable(sample_expert_actions))
    policy_loss = -fixed_log_probs.mean()
    
    if train:
        optimizer_policy.zero_grad()
        policy_loss.backward()
        '''
        ## grads are pretty big here... ~100 times bigger than values
        for par in policy_net.parameters():
            print("var: ", par)
            print("grad: ", par.grad)
        exit()
        '''
        optimizer_policy.step()
    
    return policy_loss.data.cpu().numpy()[0]

# pretrain discriminator
def pre_train_discrim(discrim_net, discrim_criterion, optimizer_discrim, size, i_iter, exp_states, exp_actions, states, actions):
    g_o_ave = 0.0
    e_o_ave = 0.0
    for _ in range(3):
        if use_gpu:
            exp_states, exp_actions, states, actions = exp_states.cuda(), exp_actions.cuda(), states.cuda(), actions.cuda()
        g_o, _ = discrim_net(Variable(states), Variable(actions))
        e_o, _ = discrim_net(Variable(exp_states), Variable(exp_actions))
        
        g_o_ave += g_o.cpu().data.mean()
        e_o_ave += e_o.cpu().data.mean()
        
        optimizer_discrim.zero_grad()
        discrim_loss = discrim_criterion(g_o, Variable(zeros((states.shape[0], states.shape[1], 1)))) + \
            discrim_criterion(e_o, Variable(ones((exp_states.shape[0], exp_states.shape[1], 1))))
        discrim_loss.backward()
        optimizer_discrim.step()
    
    if i_iter % 1 == 0:
        with open("pretrain.txt", 'a') as text_file:
            text_file.write("exp: {:.4f}\t mod: {:.4f}\n".format(e_o_ave / 3.0, g_o_ave / 3.0))
        print('exp: ', e_o_ave / 3.0, 'mod: ', g_o_ave / 3.0)
    
    return g_o_ave / 3.0

# Sampling
def collect_samples(policy_net, discrim_net, expert_states, expert_actions, size, i_iter, name="sampling", draw=False):
    # expert
    exp_ind = np.random.choice(expert_states.shape[0], size)
    sample_expert_states = torch.from_numpy(expert_states[exp_ind].copy()).transpose(0, 1)   ## seq * size * 10
    sample_expert_actions = torch.from_numpy(expert_actions[exp_ind].copy()).transpose(0, 1)
    
    # model sample
    model_states = []
    model_actions = []
    model_rewards = []
    action_means = []
    action_stds = []
    samp_ind = np.random.choice(expert_states.shape[0], size)
    mod_samples = torch.from_numpy(expert_states[samp_ind].copy()).transpose(0, 1)
    coresp_act = torch.from_numpy(expert_actions[samp_ind].copy()).transpose(0, 1) ## only for stats use
    state= mod_samples[0].clone().unsqueeze(0)  ## 1 * size * 10
    hidden_p = policy_net.init_hidden(size)
    hidden_d = discrim_net.init_hidden(size)
    if use_gpu:
        state, hidden_p, hidden_d = state.cuda(), hidden_p.cuda(), hidden_d.cuda()
    for j in range(70):
        action, hidden_p, action_mean, action_std = policy_net.select_action(Variable(state, volatile=True), hidden_p, True)
        reward, hidden_d = discrim_net.forward(Variable(state, volatile=True), action, hidden_d)
        reward = torch.log(reward).data
        action = action.data
                
        model_states.append(state.clone())
        model_actions.append(action.clone())
        action_means.append(action_mean.data.clone())
        action_stds.append(action_std.data.clone())
        
        ## env
        last_state = state.clone()
        state += action
        if use_gpu:
            state = state.cpu()
        state = state.numpy()
        
        count = np.zeros([state.shape[0], state.shape[1], 1], dtype=float)
        count[:, :, 0] = np.sum(state > 1.0, axis = 2) + np.sum(state < 0.0, axis = 2)
        count = torch.from_numpy(count)
        if use_gpu:
            count = count.cuda()
        
        state = np.clip(state, 0, 1.0)
        state = torch.from_numpy(state)
        if use_gpu:
            state = state.cuda()
        
        #reward -= count

        model_rewards.append(reward.clone())

    model_states = torch.cat(model_states, 0)
    model_actions = torch.cat(model_actions, 0)
    model_rewards = torch.cat(model_rewards, 0)
    action_means = torch.cat(action_means, 0)
    action_stds = torch.cat(action_stds, 0)
    
    ##################################
    ## Draw Data
    if draw:
        with open ("action_details.txt", 'a') as text_file:
            text_file.write('iter: {} \n'.format(i_iter))
            for i in range(action_means.shape[0]):
                text_file.write('means (): '.format(i) + to_string(action_means[i, 0, :]) + '\n')
            text_file.write('\n')
            for i in range(action_stds.shape[0]):
                text_file.write('stds (): '.format(i) + to_string(action_stds[i, 0, :]) + '\n')
            text_file.write('\n')
        mod_ave_length, mod_ave_outbound = draw_data(model_states, name, i_iter, model_actions=model_actions)
        exp_ave_length, exp_ave_outbound = draw_data(mod_samples, name + '_expert', i_iter, model_actions=coresp_act, doc_name='val_stats_expert.txt')
        
        return sample_expert_states, sample_expert_actions, model_states, model_actions, model_rewards, model_rewards.mean(), exp_ave_length, \
                exp_ave_outbound, mod_ave_length, mod_ave_outbound
    ##################################
            
    return sample_expert_states, sample_expert_actions, model_states, model_actions, model_rewards, model_rewards.mean(), 0, 0, 0, 0

# save models
def save_model(policy_net, value_net, discrim_net, path=None):
    assert path is not None
    print("saving")
    if use_gpu:
        policy_net, value_net, discrim_net = policy_net.cpu(), value_net.cpu(), discrim_net.cpu()
    pickle.dump((policy_net, value_net, discrim_net), open(path, 'wb'))
    #pickle.dump((value_net, discrim_net), open('learned_models/test_test2_more_trained_discr_gail.p', 'wb'))
    if use_gpu:
        policy_net, value_net, discrim_net = policy_net.cuda(), value_net.cuda(), discrim_net.cuda()

# compute values, advantages, action probs
def process_data(value_net, policy_net, model_states, model_actions, model_rewards, gamma, tau, i_iter, print_freq):
    if use_gpu:
        model_states, model_actions, model_rewards = model_states.cuda(), model_actions.cuda(), model_rewards.cuda()
    model_values = value_net(Variable(model_states, volatile=True))[0].data
    fixed_log_probs = policy_net.get_log_prob(Variable(model_states, volatile=True), Variable(model_actions)).data
    model_advantages, model_returns = estimate_advantages(model_rewards, model_values, gamma, tau, use_gpu)

    if i_iter % print_freq == 0:
        with open("intermediates.txt", "a") as text_file:
            text_file.write('iter: {} \n'.format(i_iter))
            text_file.write('rewards: ' + to_string(model_rewards[:, 0].squeeze()) + '\n')
            text_file.write('values: ' + to_string(model_values[:, 0].squeeze()) + '\n')
            text_file.write('advs: ' + to_string(model_advantages[:, 0].squeeze()) + '\n')
            text_file.write('returns: ' + to_string(model_returns[:, 0].squeeze()) + '\n')
            text_file.write('\n')
    
    return model_advantages, model_returns, fixed_log_probs

# transfer a 1-d vector into a string
def to_string(x):
    ret = ""
    for i in range(x.shape[0]):
        ret += "{:.3f} ".format(x[i])
    return ret

def draw_data(model_states, name, i_iter, model_actions=None, doc_name="val_stats.txt"):
    print("Drawing")
    if model_actions is not None:
        val_data = model_states.cpu().numpy()
        val_actions = model_actions.cpu().numpy()
        
        ave_stepsize = np.mean(np.abs(val_actions), axis = (0, 1))
        std_stepsize = np.std(val_actions, axis = (0, 1))
        ave_length = np.mean(np.sum(np.sqrt(np.square(val_actions[:, :, ::2]) + np.square(val_actions[:, :, 1::2])), axis = 0), axis = 0)  ## when sum along axis 0, axis 1 becomes axis 0
        ave_near_bound = np.mean((val_data < 1.0 / 100.0) + (val_data > 99.0 / 100.0), axis = (0, 1))
        
        with open(doc_name, "a") as text_file:
            text_file.write('{}_{}\n'.format(name, i_iter))
            text_file.write('ave_stepsize: ' + to_string(ave_stepsize) + '\n')
            text_file.write('std_stepsize: ' + to_string(std_stepsize) + '\n')
            text_file.write('ave_length: ' + to_string(ave_length) + '\n')
            text_file.write('ave_near_bound: ' + to_string(ave_near_bound) + '\n')
            text_file.write('\n')
    
    draw_data = model_states[:, 0, :].cpu().numpy()
    normal = [47.0, 50.0] * 5
    draw_data = draw_data * normal
    colormap = ['b', 'r', 'g', 'm', 'y', 'c']
    #plot_sequences([draw_data[:, 2:22]], macro_goals=None, colormap=colormap, save_name="imgs/{}_{}_offense+defense".format(name, i_iter), show=False, burn_in=0)
    plot_sequences([draw_data], macro_goals=None, colormap=colormap[:5], save_name="imgs/{}_{}_offense".format(name, i_iter), show=False, burn_in=0)

    if model_actions is not None:
        return np.mean(ave_length), np.mean(ave_near_bound)
    # heatmap unfinished......
    # heatmap_data = (model_states[:, :, 2:4].cpu().numpy() * [50.0, 47.0]).reshape((-1, 2))
    # plot_heatmap(heatmap_data, save_name="imgs/heatmap_{}_{}_offense".format(name, i_iter))

def load_expert_data(num):
    train_addr = "./train/"
    addrs = os.listdir(train_addr)
    Data = []
    Actions = []
    for d in addrs:
        num -= 1
        if num < 0:
            break
        
        seq_len = int(d.split('-')[2][:2])
        if seq_len != 70:
            continue
        content = open(train_addr + d, 'rb').read()
        data = np.zeros((seq_len, 22), dtype=np.float)
        action = np.zeros((seq_len-1, 22), dtype=np.float)
        for i in range(seq_len):
            pre_data = np.asarray(struct.unpack('16i', content[64*i:64*i+64]), dtype=np.float)
            for j in range(11):
                data[i, 2*j] = np.clip(pre_data[j] / 360, 0, 399) / 400
                data[i, 2*j+1] = np.clip(pre_data[j] % 360, 0, 359) / 360
        
            if i > 0:
                action[i-1] = data[i] - data[i-1]
        
        data = data[:-1]
        Data.append(data)
        Actions.append(action)

    Data = np.stack(Data)[:, :, 2:12]
    Actions = np.stack(Actions)[:, :, 2:12]
    
    tot_data = Data.shape[0]
    #rand_ind = np.random.permutation(tot_data)
    #Data, Actions = Data[rand_ind], Actions[rand_ind]
    train_data, train_action = Data[:int(tot_data*0.8)], Actions[:int(tot_data*0.8)]
    val_data, val_action = Data[int(tot_data*0.8):], Actions[int(tot_data*0.8):]
    
    ave_stepsize = np.mean(np.abs(train_action), axis = (0, 1))
    std_stepsize = np.std(train_action, axis = (0, 1))
    ave_length = np.mean(np.sum(np.sqrt(np.square(train_action[:, :, ::2]) + np.square(train_action[:, :, 1::2])), axis = 1), axis = 0)
    ave_near_bound = np.mean((train_data < 1.0 / 100.0) + (train_data > 99.0 / 100.0), axis = (0, 1))
    print(ave_stepsize, std_stepsize, ave_length, ave_near_bound)
    with open("val_stats.txt", "a") as text_file:
        text_file.write('Expert:\n')
        text_file.write('ave_stepsize: ' + to_string(ave_stepsize) + '\n')
        text_file.write('std_stepsize: ' + to_string(std_stepsize) + '\n')
        text_file.write('ave_length: ' + to_string(ave_length) + '\n')
        text_file.write('ave_near_bound: ' + to_string(ave_near_bound) + '\n')
        text_file.write('\n')
    
    print("train_data.shape:", train_data.shape, "val_data.shape:", val_data.shape)
    return train_data, train_action, val_data, val_action, ave_stepsize, std_stepsize, ave_length, ave_near_bound