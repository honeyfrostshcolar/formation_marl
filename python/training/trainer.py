import torch
import torch.nn.functional as F
import numpy as np
from collections import deque
import copy

class PPOTrainer:
    def __init__(self, model, optimizer, device='cuda', config=None):
        self.model = model
        self.old_model = copy.deepcopy(model)
        self.old_model.eval()
        self.optimizer = optimizer
        self.device = device
        
        # 配置参数
        self.config = {
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_epsilon': 0.2,
            'value_coef': 0.5,
            'entropy_coef': 0.01,
            'ppo_epochs': 4,
            'mini_batch_size': 32,
            'max_grad_norm': 0.5
        }
        if config:
            self.config.update(config)
        
        # 经验缓冲区（使用列表存储每个episode的经验）
        self.buffer = deque(maxlen=1000)  # 存储整个episode
        
        # 训练历史
        self.history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'losses': [],
            'value_losses': [],
            'policy_losses': [],
            'entropy_losses': []
        }
    
    def act(self, state, candidate_graphs):
        """
        使用旧策略选择动作
        
        Args:
            state: 包含features和robot_ids的字典
                    'features': features,
                    'positions': positions,
                    'robot_count': self.num_robots,
                    'step': self.step_count,
                    'done': False

            candidate_graphs: 候选图列表
            
        Returns:
            action: 动作字典 {graph_idx, position}
            log_prob: 动作的对数概率
            value: 状态价值
            outputs: 模型输出（用于后续存储）
        """
        with torch.no_grad():
            # 确保输入在正确的设备上
            features = state['features'].unsqueeze(0).to(self.device) if len(state['features'].shape) == 1 else state['features'].to(self.device)
            robot_count = state['robot_count']
            robot_ids = torch.arange(0, robot_count, dtype=torch.long, device=self.device)
            
            # 前向传播
            outputs = self.old_model(features, robot_ids, candidate_graphs, robot_count)
            # 'graph_scores': graph_scores,      
            # 'position_dists': position_dists,  
            # 'value': value,                   
            # 'graph_features': all_graph_features 
            
            # 采样图选择
            graph_scores = outputs['graph_scores']
            graph_probs = F.softmax(graph_scores, dim=-1)
            graph_dist = torch.distributions.Categorical(graph_probs)
            graph_idx = graph_dist.sample().item()  # 采样图索引（轮盘赌方式得到）
            
            # 采样位置
            position_dist = outputs['position_dists'][graph_idx]
            position = position_dist.sample().squeeze(0)  # [robot_count-1, 2]
            
            # 计算对数概率
            graph_log_prob = graph_dist.log_prob(torch.tensor([graph_idx], device=self.device))
            position_log_prob = position_dist.log_prob(position.unsqueeze(0)).sum()
            total_log_prob = graph_log_prob + position_log_prob
            
            # 状态价值
            value = outputs['value'].squeeze()
            
            action = {
                'graph_idx': graph_idx,
                'position': position.cpu() if self.device != 'cpu' else position
            }
            
            return action, total_log_prob.item(), value.item(), outputs
    
    def store_experience(self, state, candidate_graphs, action, log_prob, value, reward, done):
        """
        存储单步经验
        
        Args:
            state: 批量当前状态 [batch_size, state_dict]
            candidate_graphs: 候选图列表（批量共用）
            action: 批量动作 [batch_size, {graph_idx, position}]
            log_prob: 批量对数概率 [batch_size]（张量）
            value: 批量状态价值 [batch_size]（张量）
            reward: 批量奖励 [batch_size]（张量）
            done: 批量终止标志 [batch_size]（张量）
        """
        experience = {
            'state': state.copy(),
            'candidate_graphs': candidate_graphs,
            'action': action,
            'log_prob': log_prob,
            'value': value,
            'reward': reward,
            'done': done
        }
        self.buffer.append(experience)
    
    def compute_returns_and_advantages(self, last_value=0):
        """
        批量计算回报和优势函数：用PyTorch张量向量化运算，替代逐个样本循环
        Returns:
            experiences: 带回报和优势的单个样本经验列表 [总样本数]
        """
        if not self.buffer:
            return
        
        # 1. 批量提取经验，转换为PyTorch张量
        experiences = list(self.buffer)
        total_samples = len(experiences)
        
        # 提取核心数据，组成 [total_samples] 维度的张量
        rewards = torch.tensor([exp['reward'] for exp in experiences], dtype=torch.float32, device=self.device)
        values = torch.tensor([exp['value'] for exp in experiences], dtype=torch.float32, device=self.device)
        dones = torch.tensor([exp['done'] for exp in experiences], dtype=torch.float32, device=self.device)  # 0/1 张量
        
        # 2. 批量计算 next_values 和 next_nonterminals（无需循环，用切片实现）
        next_values = torch.zeros_like(values)  # [total_samples]
        next_nonterminals = 1.0 - dones  # [total_samples]（非终止标志：0=终止，1=继续）
        
        # 除了最后一个样本，其余样本的next_value = 下一个样本的value
        next_values[:-1] = values[1:]
        # 最后一个样本的next_value = 传入的last_value（终止状态为0）
        next_values[-1] = last_value
        
        # 3. 批量计算TD误差（向量化运算，一次性完成所有样本）
        gamma = self.config['gamma']
        gae_lambda = self.config['gae_lambda']
        deltas = rewards + gamma * next_values * next_nonterminals - values  # [total_samples]
        
        # 4. 批量反向计算GAE（核心：用累积乘积替代循环）
        # 先计算每个时间步的衰减系数：(gamma * gae_lambda) ^ t
        # 生成 [total_samples] 维度的衰减系数张量
        steps = torch.arange(total_samples, device=self.device)
        decay_coeffs = (gamma * gae_lambda) ** steps  # [total_samples]（从 (gammaλ)^0 到 (gammaλ)^(T-1)）
        
        # 反向累积GAE：gae[t] = delta[t] + gammaλ * next_nonterminal[t] * gae[t+1]
        # 用flip+cumprod+flip实现反向累积，避免循环
        deltas_flipped = deltas.flip(dims=[0])  # 反转deltas：[T-1, T-2, ..., 0]
        next_nonterminals_flipped = next_nonterminals.flip(dims=[0])  # 反转非终止标志
        
        # 计算累积乘积：每个位置的累积系数 = product(gammaλ * next_nonterminal[k] for k >= t)
        cum_prod = torch.cumprod(gamma * gae_lambda * next_nonterminals_flipped, dim=0)
        # 第一个位置的累积系数为1（因为没有后续步骤）
        cum_prod = torch.cat([torch.ones(1, device=self.device), cum_prod[:-1]])
        
        # 批量计算GAE：每个样本的gae = delta[t] * 累积系数[t]
        gae_flipped = deltas_flipped * cum_prod
        gae = gae_flipped.flip(dims=[0])  # 反转回原顺序：[0, 1, ..., T-1]
        
        # 5. 批量计算回报和标准化优势
        returns = gae + values  # [total_samples]（回报 = GAE + 状态价值）
        
        # 优势函数标准化（张量批量运算，比numpy更快）
        advantages = (gae - gae.mean()) / (gae.std() + 1e-8)  # [total_samples]
        
        # 6. 批量更新经验中的回报和优势（循环仅用于赋值，无计算）
        for i, exp in enumerate(experiences):
            exp['return'] = returns[i].item()
            exp['advantage'] = advantages[i].item()
        
        return experiences

    
    def update(self):
        """
        执行PPO更新
        
        Returns:
            losses_dict: 各项损失的字典
        """
        if len(self.buffer) < self.config['mini_batch_size']:
            return None
        
        # 计算回报和优势
        experiences = self.compute_returns_and_advantages()
        
        # 转换为张量
        states = []
        candidate_graphs_list = []
        actions = []
        old_log_probs = []
        advantages = []
        returns = []
        
        for exp in experiences:
            states.append(exp['state'])
            candidate_graphs_list.append(exp['candidate_graphs'])
            actions.append({
                'graph_idx': torch.tensor([exp['action']['graph_idx']], device=self.device),
                'position': exp['action']['position'].unsqueeze(0).to(self.device)
            })
            old_log_probs.append(exp['log_prob'])
            advantages.append(exp['advantage'])
            returns.append(exp['return'])
        
        # 转换为张量
        advantages = torch.tensor(advantages, device=self.device, dtype=torch.float32)
        returns = torch.tensor(returns, device=self.device, dtype=torch.float32)
        old_log_probs = torch.tensor(old_log_probs, device=self.device, dtype=torch.float32)
        
        # 多轮PPO更新
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        
        for epoch in range(self.config['ppo_epochs']):
            # 随机打乱索引
            indices = torch.randperm(len(experiences))
            
            # 小批量处理
            for start_idx in range(0, len(indices), self.config['mini_batch_size']):
                batch_indices = indices[start_idx:start_idx+self.config['mini_batch_size']]
                
                # 收集小批量数据
                batch_loss = 0
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_entropy_loss = 0
                
                for idx in batch_indices:
                    state = states[idx.item()]
                    candidate_graphs = candidate_graphs_list[idx.item()]
                    action = actions[idx.item()]
                    
                    # 重新计算旧策略输出
                    with torch.no_grad():
                        features = state['features'].unsqueeze(0).to(self.device)
                        robot_ids = state['robot_ids'].to(self.device)
                        robot_count = state['robot_count']
                        
                        old_outputs = self.old_model(features, robot_ids, candidate_graphs, robot_count)
                    
                    # 计算新策略输出
                    new_outputs = self.model(features, robot_ids, candidate_graphs, robot_count)
                    
                    # 计算图选择损失
                    old_graph_logits = old_outputs['graph_scores']
                    old_graph_probs = F.softmax(old_graph_logits, dim=-1)
                    old_graph_log_prob = torch.log(old_graph_probs[0, action['graph_idx']])
                    
                    new_graph_probs = F.softmax(new_outputs['graph_scores'], dim=-1)
                    new_graph_log_prob = torch.log(new_graph_probs[0, action['graph_idx']])
                    
                    # 计算位置损失
                    graph_idx = action['graph_idx'].item()
                    old_position_dist = old_outputs['position_dists'][graph_idx]
                    old_position_log_prob = old_position_dist.log_prob(action['position']).sum()
                    
                    new_position_dist = new_outputs['position_dists'][graph_idx]
                    new_position_log_prob = new_position_dist.log_prob(action['position']).sum()
                    
                    # 总的对数概率
                    old_total_log_prob = old_graph_log_prob + old_position_log_prob
                    new_total_log_prob = new_graph_log_prob + new_position_log_prob
                    
                    # 计算比率
                    ratio = torch.exp(new_total_log_prob - old_total_log_prob)
                    
                    # PPO clip损失
                    adv = advantages[idx]
                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1 - self.config['clip_epsilon'], 
                                       1 + self.config['clip_epsilon']) * adv
                    policy_loss = -torch.min(surr1, surr2)
                    
                    # 价值损失
                    value_pred = new_outputs['value'].squeeze()
                    return_target = returns[idx]
                    value_loss = F.mse_loss(value_pred, return_target)
                    
                    # 熵正则化
                    graph_entropy = -(new_graph_probs * torch.log(new_graph_probs + 1e-8)).sum()
                    
                    position_entropy = 0
                    for dist in new_outputs['position_dists']:
                        position_entropy += dist.entropy().mean()
                    position_entropy = position_entropy / len(new_outputs['position_dists'])
                    
                    entropy_loss = -(graph_entropy + position_entropy) * self.config['entropy_coef']
                    
                    # 总损失
                    loss = (policy_loss + 
                           self.config['value_coef'] * value_loss + 
                           entropy_loss)
                    
                    batch_loss += loss
                    batch_policy_loss += policy_loss.item()
                    batch_value_loss += value_loss.item()
                    batch_entropy_loss += entropy_loss.item()
                
                # 平均损失
                batch_size = len(batch_indices)
                avg_loss = batch_loss / batch_size
                
                # 反向传播
                self.optimizer.zero_grad()
                avg_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
                self.optimizer.step()
                
                # 累计损失
                total_loss += avg_loss.item()
                total_policy_loss += batch_policy_loss / batch_size
                total_value_loss += batch_value_loss / batch_size
                total_entropy_loss += batch_entropy_loss / batch_size
        
        # 计算平均损失
        num_updates = self.config['ppo_epochs'] * (len(indices) // self.config['mini_batch_size'] + 1)
        
        losses_dict = {
            'total_loss': total_loss / num_updates if num_updates > 0 else 0,
            'policy_loss': total_policy_loss / num_updates if num_updates > 0 else 0,
            'value_loss': total_value_loss / num_updates if num_updates > 0 else 0,
            'entropy_loss': total_entropy_loss / num_updates if num_updates > 0 else 0
        }
        
        # 清空缓冲区
        self.buffer.clear()
        
        # 更新旧模型
        self.old_model.load_state_dict(self.model.state_dict())
        
        # 记录损失
        self.history['losses'].append(losses_dict['total_loss'])
        self.history['value_losses'].append(losses_dict['value_loss'])
        self.history['policy_losses'].append(losses_dict['policy_loss'])
        self.history['entropy_losses'].append(losses_dict['entropy_loss'])
        
        return losses_dict
    
    def finish_episode(self, episode_reward, episode_length):
        """
        完成一个episode，记录结果
        
        Args:
            episode_reward: 累计奖励
            episode_length: episode长度
        """
        self.history['episode_rewards'].append(episode_reward)
        self.history['episode_lengths'].append(episode_length)
        
        # 如果缓冲区有经验，执行更新
        if len(self.buffer) > 0:
            self.update()
    
    def save_checkpoint(self, path, episode, config):
        """
        保存检查点
        
        Args:
            path: 保存路径
            episode: 当前episode数
            config: 训练配置
        """
        checkpoint = {
            'episode': episode,
            'model_state_dict': self.model.state_dict(),
            'old_model_state_dict': self.old_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': config,
            'history': self.history
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path):
        """
        加载检查点
        
        Args:
            path: 检查点路径
            
        Returns:
            episode: 加载的episode数
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.old_model.load_state_dict(checkpoint['old_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        
        # 更新配置
        if 'config' in checkpoint:
            self.config.update(checkpoint['config'])
        
        print(f"Checkpoint loaded from {path}")
        return checkpoint['episode']
    
    def get_training_stats(self):
        """获取训练统计信息"""
        if not self.history['episode_rewards']:
            return {}
        
        last_10_rewards = self.history['episode_rewards'][-10:] if len(self.history['episode_rewards']) >= 10 else self.history['episode_rewards']
        
        return {
            'current_episode': len(self.history['episode_rewards']),
            'avg_reward_last_10': np.mean(last_10_rewards) if last_10_rewards else 0,
            'max_reward': np.max(self.history['episode_rewards']) if self.history['episode_rewards'] else 0,
            'min_reward': np.min(self.history['episode_rewards']) if self.history['episode_rewards'] else 0,
            'avg_episode_length': np.mean(self.history['episode_lengths']) if self.history['episode_lengths'] else 0
        }