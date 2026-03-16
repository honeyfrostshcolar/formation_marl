import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from models.g2anet_maddpg import G2ANet_MADDPG_Actor, Centralized_Critic

class MADDPG_Agent:
    def __init__(self, num_followers, obs_dim=32, action_dim=2, lr_actor=5e-5, lr_critic=3e-4, gamma=0.99, tau=0.005):
        self.num_followers = num_followers
        self.gamma = gamma # 折扣因子
        self.tau = tau # 软更新系数
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. 初始化 Actor (小弟的 G2ANet 大脑，所有人共享这一个！)
        self.actor = G2ANet_MADDPG_Actor(action_dim=action_dim).to(self.device) # 这里的 action_dim 是每个小弟的动作维度，比如 2 (dx, dy)
        self.target_actor = G2ANet_MADDPG_Actor(action_dim=action_dim).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict()) # 初始权重同步
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        # 2. 初始化 Critic (上帝的判卷笔)
        self.critic = Centralized_Critic(num_followers, obs_dim, action_dim).to(self.device)
        self.target_critic = Centralized_Critic(num_followers, obs_dim, action_dim).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict()) # 初始权重同步
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

    def select_action(self, obs_array, add_noise=True, noise_scale=0.15):
        """
        环境交互时调用 (分布式执行)
        obs_array: 形状 (num_followers, obs_dim)
        """
        obs_tensor = torch.FloatTensor(obs_array).to(self.device)
        
        self.actor.eval() # 切换到预测模式
        with torch.no_grad():
            action_tensor, hard_weights, soft_weights = self.actor(obs_tensor)
        self.actor.train() # 切回训练模式
        
        action = action_tensor.cpu().numpy()
        graphs = hard_weights.cpu().numpy()
        graphs_soft = soft_weights.cpu().numpy()
        
        
        # MADDPG 是确定性策略，必须手动加高斯噪声来探索环境
        if add_noise:
            noise = np.random.normal(0, noise_scale, size=action.shape) # noise_scale是噪声方差，可调
            action = np.clip(action + noise, -1.0, 1.0) # 保证动作不越界
            
        return action, graphs, graphs_soft

    def update(self, sample_batch):
        """
        核心炼丹炉：计算 Loss 并更新梯度 (集中式训练)
        传入的 sample_batch 是从 ReplayBuffer 里抽出来的一批数据
        """
        # 假设抽出来的 batch_size = 64
        # obs_batch 形状: (64, num_followers, obs_dim)
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = sample_batch

        batch_size = obs_batch.size(0)

        # 把单个人的数据展平，拼成上帝视角需要的全局数据
        # 形状变为: (64, num_followers * obs_dim)
        global_obs = obs_batch.view(batch_size, -1)
        global_next_obs = next_obs_batch.view(batch_size, -1)
        global_actions = action_batch.view(batch_size, -1)

        # ------------------------------------
        # 一、 更新 Critic (让上帝的打分越来越准)
        # ------------------------------------
        with torch.no_grad():
            # 1. 让 target_actor 预测下一步的动作
            next_actions = []
            for i in range(self.num_followers):
                # 抽出第 i 个小弟的 next_obs 输入网络
                n_a, _, _ = self.target_actor(next_obs_batch[:, i, :]) 
                next_actions.append(n_a)
            # 拼成全局动作 (64, num_followers * action_dim)
            global_next_actions = torch.cat(next_actions, dim=-1) 

            # 2. 让 target_critic 评估下一步的 Q 值
            target_q = self.target_critic(global_next_obs, global_next_actions)
            
            # 3. 计算贝尔曼目标方程： y = r + gamma * Q'
            # (这里假设所有小弟拿的是同一个全局团队 reward)
            target_q_val = reward_batch + (1 - done_batch) * self.gamma * target_q

        # 计算当前 Critic 的 Q 值
        current_q = self.critic(global_obs, global_actions)

        # Critic 的 Loss 就是预测 Q 值和目标 Q 值的均方差 (MSE)
        critic_loss = F.mse_loss(current_q, target_q_val)

        # 反向传播更新 Critic
        self.critic_optimizer.zero_grad() # 清空旧的梯度
        critic_loss.backward() # 计算新的梯度
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0) # 梯度裁剪，防止爆炸
        self.critic_optimizer.step()

        # ------------------------------------
        # 二、 更新 Actor (让小弟的行为迎合上帝的高分)
        # ------------------------------------
        # 1. 让最新的 Actor 对当前状态重新做一次决策
        curr_actions = []
        for i in range(self.num_followers):
            c_a, _, _ = self.actor(obs_batch[:, i, :]) 
            curr_actions.append(c_a)
        global_curr_actions = torch.cat(curr_actions, dim=-1)

        # 2. 让 Critic 给这套新动作打分
        # 注意：这里我们是要最大化 Q 值，但 PyTorch 的优化器是“最小化”Loss
        # 所以我们在 Q 值前面加个负号！
        actor_loss = -self.critic(global_obs, global_curr_actions).mean()

        # 反向传播更新 Actor (G2ANet 的权重就在这里被更新！)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0) # 梯度裁剪，防止爆炸
        self.actor_optimizer.step()

        # ------------------------------------
        # 三、 软更新目标网络 (Soft Update)
        # ------------------------------------
        # 把当前网络的参数，以极其微小的比例 (tau) 慢慢融进 Target 网络里
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        return actor_loss.item(), critic_loss.item()