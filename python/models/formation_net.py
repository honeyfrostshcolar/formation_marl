import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from torch_geometric.nn import GATConv

class ConstrainedFormationNet(nn.Module):
    """
    带约束的编队选择神经网络
    基于分层结构和约束位置生成
    输入固定维度的环境特征向量，输出每个控制图的安全位置配置和编队选择分数
    """
    
    def __init__(self, feature_dim, num_graphs, num_robots, 
                 min_distance=0.5, max_distance=3.0, angle_range=(-90, 90)):
        super(ConstrainedFormationNet, self).__init__()
        
        self.num_graphs = num_graphs
        self.num_robots = num_robots
        self.num_followers = num_robots - 1  # 减去领航者
        
        # 约束参数
        self.min_distance = min_distance # 跟随者与领航者的最小距离
        self.max_distance = max_distance # 跟随者与领航者的最大距离
        self.angle_min, self.angle_max = angle_range # 跟随者与领航者的角度范围

        self.input_norm = nn.LayerNorm(feature_dim)
        
        ###### 初始化神经网络模块
        # 共享特征编码器
        self.encoder = nn.Sequential(
            self.input_norm,  # 先归一化输入
            nn.Linear(feature_dim, 128),  #全连接层
            nn.ReLU(),  
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # 2. GNN层（处理可变节点数）
        self.gnn_layers = nn.ModuleList([
            GATConv(4 + 32 + 16, 64, heads=2, concat=False),  # 输入: 4(机器人编码) + 32(环境) + 16(图结构) = 52
            GATConv(64, 64, heads=2, concat=False),
            GATConv(64, 64, heads=2, concat=False)
        ])
        
        # 3. 位置生成器（连续Actor）
        self.position_generator = nn.Sequential(
            nn.Linear(64, 32),  # GNN输出的节点特征
            nn.ReLU(),
            nn.Linear(32, 2)    # 每个机器人的2D位置
        )
        
        # 4. 图结构编码器（将控制图编码为特征向量）
        # 假设最大机器人数量为max_robots = 10
        self.graph_encoder = nn.Sequential(
            nn.Linear(10 * 10, 128),  # 邻接矩阵展平
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16)  # 16维图特征
        )
        
        # 5. 编队选择器（离散Actor）
        # 不再输出固定数量的分数，而是通过图匹配计算分数
        self.graph_scorer = nn.Sequential(
            nn.Linear(32 + 16, 64),  # 环境特征 + 图特征
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # 输出单个图的分数
        )
        
        # 6. 机器人ID嵌入
        self.robot_embedding = nn.Embedding(10, 4) # 机器人ID（最多为10个）嵌入为4维向量

        self.position_log_std = nn.Parameter(torch.zeros(1, 2))

        # 评估当前状态的价值（Critic） 
        self.critic = nn.Sequential(
            nn.Linear(32 + 16, 64),  # 环境特征 + 平均图特征
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # 状态价值
        )
        
        # 位置微调模块（用于后期强化学习微调）
        # self.position_refiners = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(32 + 2 * self.num_followers, 32),
        #         nn.ReLU(),
        #         nn.Linear(32, 2 * self.num_followers)
        #     ) for _ in range(num_graphs)
        # ])
        
    def forward(self, features, robot_ids, candidate_graphs, robot_count):
        """
        前向传播
        
        Args:
            env_features: 环境特征 [batch_size, feature_dim]
            training_phase: 训练阶段 ("imitation", "mixed", "rl_finetune")
            robot_count: 整数，机器人数量
            candidate_graphs: 列表，每个元素是一个邻接矩阵 [robot_count, robot_count]
                            不同机器人数量对应的图数量不同
        """
        batch_size = features.size(0) # 获取批量大小
        num_graphs = len(candidate_graphs) # 获取控制图数量

        #print("batch_size:", batch_size)
        
        # 特征编码
        env_features = self.encoder(features) # [batch_size, 32]

        graph_features_list = [] 
        for graph in candidate_graphs:
            # 将图填充到最大尺寸
            pad_size = (0, 10 - robot_count,
                       0, 10 - robot_count)
            graph_padded = F.pad(graph, pad_size, mode='constant', value=0)
            
            # 编码图结构
            graph_flat = graph_padded.flatten().unsqueeze(0)  # [1, max_robots²]
            graph_feat = self.graph_encoder(graph_flat)  # [1, 16]
            graph_features_list.append(graph_feat) # [num_graphs, 16]

        # 堆叠所有图特征 [num_graphs, 16]
        all_graph_features = torch.cat(graph_features_list, dim=0)
        # 扩展环境特征到每个图
        env_expanded = env_features.unsqueeze(1)  # [batch, 1, 32]
        env_expanded = env_expanded.expand(-1, num_graphs, -1)  # [batch, num_graphs, 32]
        
        # 扩展图特征到每个batch
        graph_expanded = all_graph_features.unsqueeze(0)  # [1, num_graphs, 16]
        graph_expanded = graph_expanded.expand(batch_size, -1, -1)  # [batch, num_graphs, 16]
        
        # 拼接特征并计算分数（combined包括了环境特征和图特征）
        combined = torch.cat([env_expanded, graph_expanded], dim=-1)  # [batch, num_graphs, 48]

        # 重塑为 [batch * num_graphs, 48] 以便批量处理
        combined_flat = combined.view(-1, 48)
        scores_flat = self.graph_scorer(combined_flat)  # [batch * num_graphs, 1]
        
        # 恢复形状
        graph_scores = scores_flat.view(batch_size, num_graphs)  # [batch, num_graphs]

        # 为每个候选图生成位置分布
        position_dists = []
        
        # 预计算机器人ID嵌入
        # follower_robot_ids = robot_ids[1:]
        robot_emb = self.robot_embedding(robot_ids)  # [batch * robot_count, 4] 
        
        # 预计算扩展的环境特征（每个机器人一份）
        env_per_robot = env_features.repeat_interleave(robot_count, dim=0)  # [batch * robot_count, 32]

        # 对每个候选图独立处理
        for i, graph in enumerate(candidate_graphs):
            # 获取当前图的特征
            graph_feat = all_graph_features[i]  # [16]
            graph_feat_per_robot = graph_feat.unsqueeze(0).expand(batch_size * robot_count, -1)  # [batch * (robot_count), 16]
            
            # 构建GNN输入
            gnn_input = torch.cat([
                robot_emb,           # [batch * (robot_count), 4]
                env_per_robot,       # [batch * (robot_count), 32]
                graph_feat_per_robot # [batch * (robot_count), 16]
            ], dim=1)  # [batch * (robot_count), 52]
            
            # GNN处理
            edge_index = self._create_edges(batch_size, robot_count, graph)
            x = gnn_input
            
            for gnn_layer in self.gnn_layers:
                x = gnn_layer(x, edge_index)
                x = F.relu(x)

            # 获取跟随者节点的特征（假设领导者是第一个机器人）
            # 重塑为 [batch, robot_count, 64]
            x_reshaped = x.view(batch_size, robot_count, -1)
            
            # 只处理跟随者（索引1到robot_count-1）
            follower_features = x_reshaped[:, 1:, :]  # [batch, robot_count-1, 64]

            # 先处理图消息，在处理位置生成会不会好一点
            
            # 生成位置均值（获取相对位置）
            position_mean = self.position_generator(follower_features)  # [batch, robot_count-1, 2]

            # 创建位置分布
            position_log_std = self.position_log_std.expand_as(position_mean)
            position_dist = torch.distributions.Normal(   # 正态分布
                position_mean, 
                torch.exp(position_log_std)
            ) 
            
            position_dists.append(position_dist)

        # 计算状态价值（Critic）
        # 使用平均图特征作为全局图上下文
        mean_graph_feat = torch.mean(all_graph_features, dim=0, keepdim=True)  # [1, 16]
        mean_graph_feat = mean_graph_feat.expand(batch_size, -1)  # [batch, 16]
        
        value_input = torch.cat([env_features, mean_graph_feat], dim=-1)  # [batch, 48]
        value = self.critic(value_input)  # [batch, 1]
        
        return {
            'graph_scores': graph_scores,      # [batch, num_graphs] - 用于离散动作选择
            'position_dists': position_dists,  # 列表，长度num_graphs，每个元素是位置分布
            'value': value,                    # [batch, 1] - 状态价值
            'graph_features': all_graph_features  # [num_graphs, 16] - 用于后续计算
        }
    
    def _adj_matrix_to_edge_index(self, adj_matrix):
        """将邻接矩阵转换为edge_index格式

           把 0/1 矩阵变成 ‘source→target’ 两行坐标
        """
        edge_list = []
        n = adj_matrix.size(0)
        for i in range(n):
            for j in range(n):
                if adj_matrix[i, j] > 0:  # 有连接
                    edge_list.append([i, j])
        return torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    def _create_edges(self, batch_size, robot_count, adj_matrix):
        """根据邻接矩阵创建边索引，支持批量处理"""
        edge_list = []
        
        # 获取邻接矩阵中的边（非零元素）
        if isinstance(adj_matrix, torch.Tensor):
            rows, cols = torch.where(adj_matrix > 0)
        else:
            # 如果adj_matrix是numpy数组
            rows, cols = torch.where(torch.tensor(adj_matrix) > 0)
        
        # 为每个batch样本创建边
        for b in range(batch_size):
            offset = b * robot_count
            for i, j in zip(rows, cols):
                edge_list.append([offset + i.item(), offset + j.item()])
        
        if len(edge_list) == 0:
            # 如果没有边，返回空张量
            return torch.empty((2, 0), dtype=torch.long, device=adj_matrix.device if hasattr(adj_matrix, 'device') else 'cpu')
        
        return torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    def _batch_edge_index(self, edge_index, num_nodes, batch_size):
        """为批次处理扩展边索引
        
           给每张图的边编号加上偏移，拼成一批大图，实现 GNN 的批量并行
        """
        edge_indices = []
        for b in range(batch_size):
            offset = b * num_nodes
            edges = edge_index + offset
            edge_indices.append(edges)
        return torch.cat(edge_indices, dim=1)

    def _polar_to_constrained_cartesian(self, polar_params):
        """
        将极坐标参数转换为带约束的直角坐标
        
        Args:
            polar_params: [batch_size, num_followers, 2]
                         [:, :, 0] 距离参数 (0-1)
                         [:, :, 1] 角度参数 (0-1)
        """
        # 应用sigmoid确保参数在0-1范围内
        distance_param = torch.sigmoid(polar_params[..., 0])
        angle_param = torch.sigmoid(polar_params[..., 1])
        
        # 转换为实际距离和角度
        distances = distance_param * (self.max_distance - self.min_distance) + self.min_distance
        angles = angle_param * (self.angle_max - self.angle_min) + self.angle_min
        
        # 转换为弧度
        angles_rad = angles * torch.pi / 180.0
        
        # 转换为直角坐标 (在领航者后方)
        x = -distances * torch.cos(angles_rad)  # 负号确保在后方
        y = distances * torch.sin(angles_rad)
        
        return torch.stack([x, y], dim=-1)
    
    def _cartesian_to_polar(self, cartesian_pos):
        """
        将直角坐标转换为极坐标参数
        并归一化到0-1范围内
        最后逆变换用于位置微调模块
        """
        x, y = cartesian_pos[..., 0], cartesian_pos[..., 1]
        
        # 计算距离和角度
        distances = torch.sqrt(x**2 + y**2)
        angles = torch.atan2(y, -x) * 180.0 / torch.pi  # 注意负号
        
        # 归一化到0-1范围
        distance_param = (distances - self.min_distance) / (self.max_distance - self.min_distance)
        angle_param = (angles - self.angle_min) / (self.angle_max - self.angle_min)
        
        # 应用sigmoid逆函数（近似）
        distance_param = torch.logit(distance_param.clamp(1e-6, 1-1e-6))
        angle_param = torch.logit(angle_param.clamp(1e-6, 1-1e-6))
        
        return torch.stack([distance_param, angle_param], dim=-1)

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