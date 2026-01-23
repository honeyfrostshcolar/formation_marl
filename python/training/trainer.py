from python.models.formation_net import SimplifiedPPOLoss
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import sys
import formation_core 
import copy
import torch.nn.functional as F


class SimplifiedPPOTrainer:
    def __init__(self, model, optimizer, device='cuda'):
        self.model = model
        self.old_model = copy.deepcopy(model)  # 用于收集经验
        self.old_model.eval()  # 旧策略不训练
        self.optimizer = optimizer
        self.device = device
        self.loss_fn = SimplifiedPPOLoss()
        
        # 经验缓冲区
        self.buffer = []
    
    def collect_experience(self, state, candidate_graphs, action, reward, done, next_state=None):
        """
        收集单步经验
        
        Args:
            state: 当前状态
            candidate_graphs: 候选图列表
            action: 执行的动作 {graph_idx, position}
            reward: 获得的奖励
            done: 是否结束
            next_state: 下一状态（可选）
        """
        with torch.no_grad():
            # 使用旧策略计算价值和对数概率
            outputs = self.old_model(
                state['features'].unsqueeze(0).to(self.device),
                state['robot_ids'].to(self.device),
                candidate_graphs,
                state['robot_count']
            )
            
            # 获取状态价值
            value = outputs['value'].squeeze()
            
            # 获取选择动作的对数概率
            # 图选择的对数概率
            graph_probs = F.softmax(outputs['graph_scores'], dim=-1)
            graph_log_prob = torch.log(graph_probs[0, action['graph_idx']])
            
            # 位置生成的对数概率
            position_dist = outputs['position_dists'][action['graph_idx'].item()]
            position_log_prob = position_dist.log_prob(action['position'].unsqueeze(0)).sum()
            
            total_log_prob = graph_log_prob + position_log_prob
            
            # 存储经验
            experience = {
                'state': state,
                'candidate_graphs': candidate_graphs,
                'action': action,
                'reward': reward,
                'value': value.item(),
                'log_prob': total_log_prob.item(),
                'done': done,
                'next_state': next_state
            }
            
            self.buffer.append(experience)
    
    def compute_gae(self, gamma=0.99, gae_lambda=0.95):
        """计算GAE优势函数"""
        if not self.buffer:
            return [], []
        
        advantages = []
        returns = []
        
        # 计算每个时间步的回报和优势
        last_value = 0  # 如果是终止状态，下一状态价值为0
        
        for t in reversed(range(len(self.buffer))):
            reward = self.buffer[t]['reward']
            value = self.buffer[t]['value']
            done = self.buffer[t]['done']
            
            # 如果是终止状态，下一状态价值为0
            next_value = 0 if (t == len(self.buffer)-1 or done) else self.buffer[t+1]['value']
            
            # TD误差
            delta = reward + gamma * next_value * (1 - done) - value
            
            # GAE
            if t == len(self.buffer) - 1:
                gae = delta
            else:
                gae = delta + gamma * gae_lambda * (1 - done) * advantages[0]
            
            advantages.insert(0, gae)
            returns.insert(0, gae + value)
        
        return advantages, returns
    
    def update(self, advantages, returns, ppo_epochs=4, batch_size=None):
        """执行PPO更新"""
        if not self.buffer:
            return None
        
        num_samples = len(self.buffer)
        if batch_size is None or batch_size > num_samples:
            batch_size = num_samples
        
        # 转换为张量
        advantages = torch.tensor(advantages, device=self.device).float()
        returns = torch.tensor(returns, device=self.device).float()
        
        # 多轮PPO更新
        for epoch in range(ppo_epochs):
            # 随机打乱
            indices = torch.randperm(num_samples)
            
            for start_idx in range(0, num_samples, batch_size):
                batch_indices = indices[start_idx:start_idx+batch_size]
                
                # 准备批量数据
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # 收集批量数据
                batch_states = []
                batch_candidate_graphs = []
                batch_actions = {'graph_idx': [], 'position': []}
                
                for idx in batch_indices:
                    exp = self.buffer[idx]
                    batch_states.append(exp['state'])
                    batch_candidate_graphs.append(exp['candidate_graphs'])
                    batch_actions['graph_idx'].append(torch.tensor([exp['action']['graph_idx']], device=self.device))
                    batch_actions['position'].append(exp['action']['position'].to(self.device))
                
                # 由于每个样本的候选图数量可能不同，需要逐个处理
                total_loss = 0
                loss_count = 0
                
                for i in range(len(batch_indices)):
                    state = batch_states[i]
                    candidate_graphs = batch_candidate_graphs[i]
                    
                    # 重新计算旧策略输出
                    with torch.no_grad():
                        old_outputs = self.old_model(
                            state['features'].unsqueeze(0).to(self.device),
                            state['robot_ids'].to(self.device),
                            candidate_graphs,
                            state['robot_count']
                        )
                    
                    # 计算新策略输出
                    new_outputs = self.model(
                        state['features'].unsqueeze(0).to(self.device),
                        state['robot_ids'].to(self.device),
                        candidate_graphs,
                        state['robot_count']
                    )
                    
                    # 准备动作
                    action = {
                        'graph_idx': batch_actions['graph_idx'][i],
                        'position': batch_actions['position'][i].unsqueeze(0)
                    }
                    
                    # 计算损失
                    loss, _ = self.loss_fn(
                        new_outputs,
                        old_outputs,
                        action,
                        batch_advantages[i:i+1],
                        batch_returns[i:i+1]
                    )
                    
                    total_loss += loss
                    loss_count += 1
                
                # 平均损失并反向传播
                if loss_count > 0:
                    avg_loss = total_loss / loss_count
                    
                    self.optimizer.zero_grad()
                    avg_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                    self.optimizer.step()
        
        # 清空缓冲区
        self.buffer.clear()
        
        # 更新旧模型
        self.old_model.load_state_dict(self.model.state_dict())
        
        return avg_loss.item() if loss_count > 0 else None