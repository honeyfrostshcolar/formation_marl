import torch
import torch.nn as nn
import torch.nn.functional as F

class SimplifiedPPOLoss(nn.Module):
    def __init__(self, clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.01):
        super().__init__()
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
    
    def forward(self, new_outputs, old_outputs, actions, advantages, returns):
        """
        计算PPO损失
        
        Args:
            new_outputs: 新策略的输出
            old_outputs: 旧策略的输出（结构相同）
            actions: 包含图索引和位置的动作
                graph_idx: [batch] 选择的图索引
                position: [batch, robot_count-1, 2] 选择的位置
            advantages: [batch] 优势函数（动作比当前状态的平均水平好多少）
            returns: [batch] 回报
        """
        batch_size = actions['graph_idx'].size(0)
        
        # 1. 离散动作损失（图选择）
        # 获取新旧策略的图选择概率
        new_graph_logits = new_outputs['graph_scores']
        old_graph_logits = old_outputs['graph_scores']
        
        # 转换为概率分布
        new_graph_probs = F.softmax(new_graph_logits, dim=-1)
        old_graph_probs = F.softmax(old_graph_logits, dim=-1)
        
        # 获取选择动作的概率（从概率表中精准找出对应图的概率值）
        new_graph_log_probs = torch.log(new_graph_probs.gather(1, actions['graph_idx'].unsqueeze(1))).squeeze(1) #新策略下，“选中该图” 的对数概率
        old_graph_log_probs = torch.log(old_graph_probs.gather(1, actions['graph_idx'].unsqueeze(1))).squeeze(1) #旧策略下，“选中该图” 的对数概率
        
        # 计算比率和PPO clip损失
        ratio_graph = torch.exp(new_graph_log_probs - old_graph_log_probs)
        graph_surr1 = ratio_graph * advantages
        graph_surr2 = torch.clamp(ratio_graph, 1-self.clip_epsilon, 1+self.clip_epsilon) * advantages
        graph_loss = -torch.min(graph_surr1, graph_surr2).mean()
        
        # 2. 连续动作损失（位置生成）
        position_loss = 0
        ratio_position_list = []
        
        # 对每个样本单独处理
        for i in range(batch_size):
            graph_idx = actions['graph_idx'][i].item()
            
            # 获取该图对应的位置分布
            new_position_dist = new_outputs['position_dists'][graph_idx]
            old_position_dist = old_outputs['position_dists'][graph_idx]
            
            # 获取该样本的位置动作
            sample_position = actions['position'][i:i+1]  # [1, robot_count-1, 2]
            
            # 计算新旧策略下该位置的对数概率
            new_position_log_prob = new_position_dist.log_prob(sample_position).sum(dim=[1, 2])
            old_position_log_prob = old_position_dist.log_prob(sample_position).sum(dim=[1, 2])
            
            # 计算比率
            ratio_position = torch.exp(new_position_log_prob - old_position_log_prob)
            ratio_position_list.append(ratio_position)
            
            # PPO clip损失
            position_surr1 = ratio_position * advantages[i]
            position_surr2 = torch.clamp(ratio_position, 1-self.clip_epsilon, 1+self.clip_epsilon) * advantages[i]
            position_loss = position_loss - torch.min(position_surr1, position_surr2)
        
        position_loss = position_loss / batch_size
        
        # 3. 价值损失
        value = new_outputs['value'].squeeze(-1)
        value_loss = F.mse_loss(value, returns)
        
        # 4. 熵正则化
        # 图选择的熵（鼓励探索）
        graph_entropy = -(new_graph_probs * torch.log(new_graph_probs + 1e-8)).sum(dim=-1).mean()
        
        # 位置生成的熵（所有位置分布的平均熵）
        position_entropy = 0
        for dist in new_outputs['position_dists']:
            position_entropy += dist.entropy().mean()
        position_entropy = position_entropy / len(new_outputs['position_dists'])
        
        entropy_bonus = graph_entropy + position_entropy
        
        # 5. 总损失
        total_loss = (
            graph_loss + 
            position_loss + 
            self.value_coef * value_loss - 
            self.entropy_coef * entropy_bonus
        )
        
        # 记录各项损失
        losses_dict = {
            'total_loss': total_loss.item(),
            'graph_loss': graph_loss.item(),
            'position_loss': position_loss.item(),
            'value_loss': value_loss.item(),
            'graph_entropy': graph_entropy.item(),
            'position_entropy': position_entropy.item(),
            'graph_ratio_mean': ratio_graph.mean().item(),
            'position_ratio_mean': torch.mean(torch.stack(ratio_position_list)).item() if ratio_position_list else 0
        }
        
        return total_loss, losses_dict

    
    def _compute_diversity_loss(self, positions):
        """计算位置配置的多样性损失"""
        batch_size, num_graphs, num_robots, _ = positions.shape
        
        if num_graphs < 2:
            return torch.tensor(0.0, device=positions.device)
        
        total_distance = 0.0
        count = 0
        
        # 计算所有控制图对之间的平均距离
        for i in range(num_graphs):
            for j in range(i + 1, num_graphs):
                pos_i = positions[:, i]  # [batch_size, num_robots, 2]
                pos_j = positions[:, j]
                
                # 计算平均位置差异
                diff = pos_i - pos_j
                distance = torch.mean(torch.norm(diff, dim=2))
                total_distance += distance
                count += 1
        
        if count == 0:
            return torch.tensor(0.0, device=positions.device)
        
        avg_distance = total_distance / count
        
        # 我们希望平均距离不要太接近0（即不要所有控制图都生成相似的位置）
        # 因为模型的优化逻辑是 “最小化总损失”，所以损失函数必须满足：avg_distance 与损失值呈 “负相关”
        diversity_loss = -torch.log(avg_distance + 1e-8)
        
        return diversity_loss
    
def sample_and_decode_action(model_output, robot_count, training=True, explore_rate=0.1, pos_scale=5.0):
    """
        模型输出转环境step可用的action
        
    Args:
        model_output: 模型forward的返回字典
        robot_count: 机器人总数（1领航者+N跟随者）
        training: 是否训练模式（训练=随机采样，测试=贪心选择）
        explore_rate: 训练时的探索概率（随机选择graph_idx）
        pos_scale: 位置归一化缩放因子（对应step里的*5.0）
        Returns:
            action: 环境step可直接接收的列表 [graph_idx_norm, x1, y1, x2, y2...]
            sample_info: 采样信息（用于训练时计算损失）
    """
    # 1. 提取模型输出
    graph_scores = model_output['graph_scores']  # [batch, num_graphs]
    position_dists = model_output['position_dists']  # 列表，长度num_graphs
    num_followers = robot_count - 1
        
    # 2. 采样graph_idx并归一化到[-1,1]
    graph_probs = F.softmax(graph_scores, dim=-1)
    if training and torch.rand(1).item() < explore_rate:
        graph_idx = torch.randint(0, graph_probs.shape[1], (graph_probs.shape[0],))
    else:
        graph_idx = torch.argmax(graph_probs, dim=-1)
    num_graphs = graph_probs.shape[1]
    graph_idx_norm = (graph_idx / (num_graphs - 1)) * 2 - 1  # 转[-1,1]
        
    # 3. 采样位置并归一化展平
    pos_dist = position_dists[graph_idx.item()]  # 取对应图的位置分布
    follower_pos = pos_dist.sample()  # [batch, num_followers, 2]
    follower_pos_norm = follower_pos / pos_scale  # 归一化到[-1,1]
    follower_pos_flat = follower_pos_norm.flatten().tolist()
        
    # 4. 拼接成step的action（仅取batch_size=1的情况，适配单环境交互）
    action = [graph_idx_norm.item()] + follower_pos_flat # 这里的动作暂时不对
        
    # 5. 记录采样信息（训练时用于损失计算）
    sample_info = {
        'graph_idx': graph_idx,  # [batch,]
        'graph_log_prob': torch.log(graph_probs.gather(1, graph_idx.unsqueeze(1))).squeeze(1), # 图选择对数概率
        'position': follower_pos,  # [batch, num_followers, 2]
        'position_log_prob': pos_dist.log_prob(follower_pos).sum(dim=[1,2]) # 位置选择对数概率
    }
        
    return action, sample_info
