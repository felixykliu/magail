import argparse
import os
import sys
import pickle
import time
import numpy as np
import shutil
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *
from models.mlp_policy import Policy
from models.mlp_critic import Value
from models.mlp_policy_disc import DiscretePolicy
from models.mlp_discriminator import Discriminator
from torch.autograd import Variable
from torch import nn
from core.ppo import ppo_step
from core.trpo import trpo_step

from helpers import *
import visdom

Tensor = DoubleTensor
torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch GAIL example')
parser.add_argument('--log-std', type=float, default=0, metavar='G',
                    help='log std for the policy (default: 0)')
parser.add_argument('--gamma', type=float, default=0.98, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.99, metavar='G',
                    help='gae (default: 0.95)')
parser.add_argument('--l2-reg', type=float, default=1e-8, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--learning-rate', type=float, default=1e-3, metavar='G',
                    help='gae (default: 3e-4)')
parser.add_argument('--clip-epsilon', type=float, default=0.1, metavar='N',
                    help='clipping epsilon for PPO')
parser.add_argument('--num-threads', type=int, default=1, metavar='N',
                    help='number of threads for agent (default: 1)')
parser.add_argument('--seed', type=int, default=519, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--min-batch-size', type=int, default=64, metavar='N',
                    help='minimal batch size per PPO update (default: 128)')
parser.add_argument('--max-iter-num', type=int, default=8000, metavar='N',
                    help='maximal number of main iterations (default: 2000)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--draw-interval', type=int, default=10, metavar='N',
                    help='interval between drawing and more detailed information (default: 10)')
parser.add_argument('--save-model-interval', type=int, default=50, metavar='N',
                    help="interval between saving model (default: 0, means don't save)")
parser.add_argument('--ppo-epochs', type=int, default=1, metavar='N',
                    help="ppo training epochs (default: 1)")
parser.add_argument('--ppo-batch-size', type=int, default=64, metavar='N',
                    help="ppo training batch size (default: 64)")
parser.add_argument('--val-freq', type=int, default=250, metavar='N',
                    help="pretrain validation frequency (default: 250)")
parser.add_argument('--pretrain-policy-iter', type=int, default=5, metavar='N',
                    help="pretrain policy iteration (default: 5000)")
parser.add_argument('--pretrain-disc-iter', type=int, default=5, metavar='N',
                    help="pretrain discriminator iteration (default: 30)")

args = parser.parse_args()
use_gpu = True
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if use_gpu:
    torch.cuda.manual_seed_all(args.seed)

is_disc_action = False
action_dim = 10
ActionTensor = DoubleTensor

"""define actor, critic and discrimiator"""
policy_net = Policy(10, 256, 10, num_layers=2)
value_net = Value(10, 256, num_layers=3)
discrim_net = Discriminator(10, 256, 10, num_layers=3)
discrim_criterion = nn.BCELoss()

#####################################################
### Load Models
load_models = True
if load_models:
    print("Loading Models")
    policy_net, value_net, discrim_net = pickle.load(open('learned_models/nextaction_pretrain_sigpolicy.p', 'rb'))
    #_, _, discrim_net = pickle.load(open('learned_models/nextaction_trained_sigpolicy.p', 'rb'))
    print("Loading Models Finished")
#####################################################

if use_gpu:
    policy_net = policy_net.cuda()
    value_net = value_net.cuda()
    discrim_net = discrim_net.cuda()
    discrim_criterion = discrim_criterion.cuda()

optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=args.learning_rate)
optimizer_value = torch.optim.Adam(value_net.parameters(), lr=args.learning_rate)
optimizer_discrim = torch.optim.Adam(discrim_net.parameters(), lr=args.learning_rate)

# stats
vis = visdom.Visdom()
ave_rewards = []
win_ave_rewards = None
exp_p = []
win_exp_p = None
mod_p = []
win_mod_p = None
value_loss = []
win_value_loss = None
win_pre_policy = None
win_path_length = None
win_out_of_bound = None
if os.path.exists('imgs'):
    shutil.rmtree('imgs')
if not os.path.exists('imgs'):
    os.makedirs('imgs')
with open("intermediates.txt", "w") as text_file:
    text_file.write("\n")
with open("training.txt", "w") as text_file:
    text_file.write("\n")
with open("action_details.txt", "w") as text_file:
    text_file.write("\n")
with open("pretrain.txt", "w") as text_file:
    text_file.write("\n")
with open("val_stats.txt", "w") as text_file:
    text_file.write("\n")
with open("val_stats_expert.txt", "w") as text_file:
    text_file.write("\n")

# load trajectory
train_states, train_actions, val_states, val_actions, exp_ave_stepsize, exp_std_stepsize, exp_ave_length, exp_ave_near_bound = load_expert_data(40000)

# Pretrain policy
for i in range(args.pretrain_policy_iter):
    train_loss = pre_train_policy(policy_net, optimizer_policy, train_states, train_actions, args.min_batch_size)
    if i % args.val_freq == 0:
        val_loss = pre_train_policy(policy_net, optimizer_policy, val_states, val_actions, args.min_batch_size, train=False)
        collect_samples(policy_net, discrim_net, train_states, train_actions, 1, i, name="pretrain_train", draw=True)
        collect_samples(policy_net, discrim_net, val_states, val_actions, args.min_batch_size, i, name="pretrain_val", draw=True)
        
        print("Policy Pre_train Loss: {}, val_loss: {}".format(train_loss, val_loss))
        with open("pretrain.txt", 'a') as text_file:
            text_file.write("Policy Pre_train Loss: {}, val_loss: {}\n".format(train_loss, val_loss))
        update = 'append'
        if win_pre_policy is None:
            update = None
        #win_pre_policy = vis.line(X = np.array([i // args.val_freq]), Y = np.array([train_loss]), win = win_pre_policy, update = update)
        win_pre_policy = vis.line(X = np.array([i // args.val_freq]), Y = np.column_stack((np.array([val_loss]), np.array([train_loss]))), win = win_pre_policy, update = update, \
                          opts=dict(legend=['in-sample loss', 'out-of-sample loss'], title="pretrain policy training curve"))

optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=args.learning_rate * 0.5)

# Pretrain Discriminator
for i in range(args.pretrain_disc_iter):
    exp_states, exp_actions, model_states, model_actions, model_rewards, _, _, _, _, _ = \
        collect_samples(policy_net, discrim_net, train_states, train_actions, args.min_batch_size, i, draw=False)
    ret = pre_train_discrim(discrim_net, discrim_criterion, optimizer_discrim, args.min_batch_size, i, exp_states, exp_actions, model_states, model_actions)
    if ret < 0.25:
        break

optimizer_discrim = torch.optim.Adam(discrim_net.parameters(), lr=args.learning_rate * 0.05)

# Save pretrained model
if args.pretrain_policy_iter > 250:
    save_model(policy_net, value_net, discrim_net, 'learned_models/nextaction_pretrain_sigpolicy.p')

# GAN training
update_discrim = False
for i_iter in range(args.max_iter_num):
    ts0 = time.time()
    print("Collecting Data")
    exp_states, exp_actions, model_states, model_actions, model_rewards, ave_r, exp_ave_length, exp_ave_outbound, mod_ave_length, mod_ave_outbound = \
        collect_samples(policy_net, discrim_net, train_states, train_actions, args.min_batch_size, i_iter, draw = (i_iter % args.draw_interval == 0))
    
    if exp_ave_length > 0:
        update = 'append'
        if i_iter == 0:
            update = None
        win_path_length = vis.line(X = np.array([i_iter // args.draw_interval]), Y = np.column_stack((np.array([exp_ave_length]), np.array([mod_ave_length]))), win = win_path_length, \
                update = update, opts=dict(legend=['expert', 'model'], title="average path length"))
        win_out_of_bound = vis.line(X = np.array([i_iter // args.draw_interval]), Y = np.column_stack((np.array([exp_ave_outbound]), np.array([mod_ave_outbound]))), \
                win = win_out_of_bound, update = update, opts=dict(legend=['expert', 'model'], title="average out of bound rate")) 
    
    ave_rewards.append(ave_r)
    model_advantages, model_returns, fixed_log_probs = \
        process_data(value_net, policy_net, model_states, model_actions, model_rewards, args.gamma, args.tau, i_iter, args.draw_interval)
    print("Collecting Data Finished")
    ts1 = time.time()

    t0 = time.time()
    # update discriminator and value function
    if i_iter > 5 and len(mod_p) > 0:
        if update_discrim == True:
            if mod_p[-1] < 0.05:
                update_discrim = False
        else:
            if mod_p[-1] > 0.05:
                update_discrim = True
    else:
        update_discrim = False
    print("update_discrim:", update_discrim)
    
    mod_p_epoch, exp_p_epoch, value_loss_epoch = update_dis_and_critic(discrim_net, optimizer_discrim, discrim_criterion, value_net, optimizer_value, exp_states, exp_actions,\
                    model_states, model_actions, model_returns, args.l2_reg, i_iter, dis_times=3.0, critic_times=10.0, use_gpu=use_gpu, update_discrim=update_discrim)
    exp_p.append(exp_p_epoch)
    mod_p.append(mod_p_epoch)
    value_loss.append(value_loss_epoch)
    
    # update policy network using ppo
    if i_iter > 5 and mod_p[-1] < 0.8:
        for i in range(3):
            update_policy_ppo(policy_net, optimizer_policy, model_states, model_actions, model_advantages, fixed_log_probs, i_iter, args.clip_epsilon, use_gpu)
    
    t1 = time.time()

    if i_iter % args.log_interval == 0:
        print('{}\tT_sample {:.4f}\tT_update {:.4f}\tR_avg {:.3f}\texp_p {:.3f}\tmod_p {:.3f}'.format(
            i_iter, ts1-ts0, t1-t0, ave_rewards[-1], exp_p[-1], mod_p[-1]))
        
        update = 'append'
        if win_ave_rewards is None:
            update = None
        win_ave_rewards = vis.line(X = np.array([len(ave_rewards)-1]), Y = np.array([ave_rewards[-1]]), win = win_ave_rewards, update = update, \
                          opts=dict(title="training ave_rewards"))
        win_exp_p = vis.line(X = np.array([len(ave_rewards)-1]), Y = np.column_stack((np.array([exp_p[-1]]), np.array([mod_p[-1]]))), win = win_exp_p, \
                          update = update, opts=dict(legend=['expert_prob', 'model_prob'], title="training curve probs"))
        #win_mod_p = vis.line(X = np.array([len(ave_rewards)-1]), Y = np.array([mod_p[-1]]), win = win_mod_p, update = update)
        win_value_loss = vis.line(X = np.array([len(ave_rewards)-1]), Y = np.array([value_loss[-1]]), win = win_value_loss, update = update, \
                          opts=dict(title="Value Function Loss"))
        
        with open("training.txt", "a") as text_file:
            text_file.write('{}\tT_sample {:.4f}\tT_update {:.4f}\tR_avg {:.3f}\texp_p {:.3f}\tmod_p {:.3f}\n'.format(
            i_iter, ts1-ts0, t1-t0, ave_rewards[-1], exp_p[-1], mod_p[-1]))

    if args.save_model_interval > 0 and (i_iter) % args.save_model_interval == 0:
       save_model(policy_net, value_net, discrim_net, 'learned_models/nextaction_trained_sigpolicy.p')
