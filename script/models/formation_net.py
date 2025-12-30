import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
        
        ###### 初始化神经网络模块
        # 共享特征编码器
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, 128),  #全连接层
            nn.ReLU(),  
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # 位置生成分支 - 为每个控制图生成跟随者位置
        self.position_generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, 2 * self.num_followers)  # 输出极坐标参数
            ) for _ in range(num_graphs)
        ])
        
        # 编队选择分支
        self.formation_selector = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_graphs)
        )
        
        # 位置微调模块（用于后期强化学习微调）
        self.position_refiners = nn.ModuleList([
            nn.Sequential(
                nn.Linear(32 + 2 * self.num_followers, 32),
                nn.ReLU(),
                nn.Linear(32, 2 * self.num_followers)
            ) for _ in range(num_graphs)
        ])
        
    def forward(self, x, training_phase="imitation"):
        """
        前向传播
        
        Args:
            x: 环境特征 [batch_size, feature_dim]
            training_phase: 训练阶段 ("imitation", "mixed", "rl_finetune")
        """
        batch_size = x.size(0) # 获取批量大小
        
        # 特征编码
        encoded = self.encoder(x) # [batch_size, 32]
        
        # 基础位置生成
        base_positions = []
        for i, generator in enumerate(self.position_generators):
            # 通过神经网络生成极坐标参数 [batch_size, num_followers, 2]
            polar_params = generator(encoded).view(batch_size, self.num_followers, 2)
            
            # 应用约束转换为直角坐标
            positions = self._polar_to_constrained_cartesian(polar_params) # [batch_size, num_followers, 2]
            base_positions.append(positions) # [num_graphs, batch_size, num_followers, 2]
        
        //2025/12/24

        # 根据训练阶段决定是否使用微调
        if training_phase == "rl_finetune":
            refined_positions = []
            for i, (base_pos, refiner) in enumerate(zip(base_positions, self.position_refiners)):
                # 将基础位置和编码特征结合进行微调
                # refiner_input是[batch_size, 32 + num_followers*2]
                refiner_input = torch.cat([encoded, base_pos.view(batch_size, -1)], dim=1) # base_pos.view(batch_size, -1)是[batch_size, num_followers*2]
                delta_polar = refiner(refiner_input).view(batch_size, self.num_followers, 2) # [batch_size, num_followers, 2]
                
                # 应用小幅度调整（限制调整幅度）
                delta_polar = torch.tanh(delta_polar) * 0.1  # 限制在±0.1范围内
                refined_polar = self._cartesian_to_polar(base_pos) + delta_polar
                
                # 重新应用约束
                refined_pos = self._polar_to_constrained_cartesian(refined_polar) ## 应用约束转换为直角坐标
                refined_positions.append(refined_pos)
            
            all_positions = refined_positions
        else:
            all_positions = base_positions #[]
        
        # 编队选择分数
        # 这个是从神经网络得到的分数，并不是计算的评估分数
        formation_scores = self.formation_selector(encoded) # 这个只是控制图的分数 [batch_size, num_graphs]
        
        # 堆叠所有位置配置 
        positions_tensor = torch.stack(all_positions, dim=1) # [batch_size, num_graphs, num_followers, 2]
        
        # 添加领航者位置 (0, 0)
        leader_positions = torch.zeros(batch_size, self.num_graphs, 1, 2, device=x.device)
        full_positions = torch.cat([leader_positions, positions_tensor], dim=2)
        
        return full_positions, formation_scores
    
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

class HybridLoss(nn.Module):
    """
    混合损失函数：结合模仿学习和强化学习
    """
    
    def __init__(self, imitation_weight=0.7, rl_weight=0.3, diversity_weight=0.1):
        super(HybridLoss, self).__init__()
        self.imitation_weight = imitation_weight
        self.rl_weight = rl_weight
        self.diversity_weight = diversity_weight
        
        self.mse_loss = nn.MSELoss() # 均方误差损失（预测位置与专家位置的差异）连续
        self.ce_loss = nn.CrossEntropyLoss() # 交叉熵损失（预测编队分数与专家选择的差异）离散
    
    def forward(self, pred_positions, pred_scores, expert_positions, 
                expert_graph_idx, advantages, has_expert_mask):
        """
        计算混合损失
        
        Args:
            pred_positions: 预测的位置 [batch_size, num_graphs, num_robots, 2]
            pred_scores: 预测的编队分数 [batch_size, num_graphs]
            expert_positions: 专家位置 [batch_size, num_robots, 2]
            expert_graph_idx: 专家选择的编队索引 [batch_size]
            advantages: 优势函数 [batch_size, num_graphs]
            has_expert_mask: 是否有专家标注 [batch_size]
        """
        batch_size = pred_positions.size(0)
        
        # 1. 模仿学习损失（仅对有专家标注的样本）
        imitation_loss = 0.0
        if has_expert_mask.any():
            # 位置损失
            expert_selected_positions = pred_positions[has_expert_mask, expert_graph_idx[has_expert_mask]]
            position_loss = self.mse_loss(expert_selected_positions, expert_positions[has_expert_mask])
            
            # 分类损失（控制图选择）
            classification_loss = self.ce_loss(pred_scores[has_expert_mask], expert_graph_idx[has_expert_mask])
            
            imitation_loss = position_loss + classification_loss
        
        # 2. 强化学习损失（对所有样本）但是这个pred_scores指的是控制图的分数，不是位置
        # 目前由于位置是连续的，所以强化学习损失只作用于控制图选择上，之后可以尝试一下用其他强化学习方法来优化位置（比如PPO）
        log_probs = F.log_softmax(pred_scores, dim=1) # 获取对数概率
        rl_loss = -torch.mean(log_probs * advantages)
        
        # 3. 多样性损失（鼓励不同控制图生成不同的位置）
        diversity_loss = self._compute_diversity_loss(pred_positions)
        
        # 加权组合
        total_loss = (self.imitation_weight * imitation_loss + 
                     self.rl_weight * rl_loss + 
                     self.diversity_weight * diversity_loss)
        
        return total_loss, {
            'imitation_loss': imitation_loss,
            'rl_loss': rl_loss,
            'diversity_loss': diversity_loss,
            'total_loss': total_loss
        }
    
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