import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from models.magic_maddpg import MAGIC_Actor, Centralized_Critic

class MADDPG_Agent:
    def __init__(self, args):
        self.args = args
        self.num_followers = args.num_followers
        self.gamma = args.gamma # 折扣因子
        self.tau = args.tau # 软更新系数
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. 初始化 Actor (小弟的 MAGIC_Actor 大脑，所有人共享这一个！

        self.actor = MAGIC_Actor(args).to(self.device)
        self.target_actor = MAGIC_Actor(args).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict()) # 初始权重同步
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.lr_actor)

        # 2. 初始化 Critic (上帝的判卷笔)
        self.critic = Centralized_Critic(args.num_followers, args.obs_size, args.action_dim).to(self.device)
        self.target_critic = Centralized_Critic(args.num_followers, args.obs_size, args.action_dim).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict()) # 初始权重同步
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.lr_critic)

    def init_hidden(self):
        """回合开始时调用，初始化全零的记忆"""
        h = torch.zeros(1, self.num_followers, self.args.hid_size).to(self.device)
        c = torch.zeros(1, self.num_followers, self.args.hid_size).to(self.device)
        return h, c

    def select_action(self, obs_array, h_in, c_in, add_noise=True, noise_scale=0.15):
        obs_tensor = torch.FloatTensor(obs_array).unsqueeze(0).to(self.device)
        
        self.actor.eval() 
        with torch.no_grad():
            # 包装成列表 [obs, extras] 喂入
            x = [obs_tensor, (h_in, c_in)]
            # 接收返回的5个参数
            action_tensor, hard_weights, soft_weights, (h_out, c_out) = self.actor(x)
        self.actor.train() 
        
        # squeeze 剥离 dummy 的 batch 维度
        action = action_tensor.squeeze(0).cpu().numpy()
        graphs = hard_weights.squeeze(0).cpu().numpy()
        graphs_soft = soft_weights.squeeze(0).cpu().numpy()
        
        if add_noise:
            noise = np.random.normal(0, noise_scale, size=action.shape) 
            action = np.clip(action + noise, -1.0, 1.0) 
            
        return action, graphs, graphs_soft, h_out, c_out

    def update(self, sample_batch):
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch, h_in_batch, c_in_batch, h_out_batch, c_out_batch = sample_batch
        batch_size = obs_batch.size(0)

        global_obs = obs_batch.view(batch_size, -1)
        global_next_obs = next_obs_batch.view(batch_size, -1)
        global_actions = action_batch.view(batch_size, -1)

        # ------------------------------------
        # 一、 更新 Critic
        # ------------------------------------
        with torch.no_grad():
            # 整个 batch 一起喂，抛弃 for i in range(num_followers) 这种毒药写法
            x_next = [next_obs_batch, (h_out_batch, c_out_batch)]
            next_actions, _, _, _ = self.target_actor(x_next)
            
            global_next_actions = next_actions.view(batch_size, -1) 
            target_q = self.target_critic(global_next_obs, global_next_actions)
            target_q_val = reward_batch + (1 - done_batch) * self.gamma * target_q

        current_q = self.critic(global_obs, global_actions)
        critic_loss = F.mse_loss(current_q, target_q_val)

        self.critic_optimizer.zero_grad() 
        critic_loss.backward() 
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0) 
        self.critic_optimizer.step()

        # ------------------------------------
        # 二、 更新 Actor
        # ------------------------------------
        x_curr = [obs_batch, (h_in_batch, c_in_batch)]
        curr_actions, _, _, _ = self.actor(x_curr)
        global_curr_actions = curr_actions.view(batch_size, -1)

        actor_loss = -self.critic(global_obs, global_curr_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0) 
        self.actor_optimizer.step()

        # 三、 软更新
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        return actor_loss.item(), critic_loss.item()