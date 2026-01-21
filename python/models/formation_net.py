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
            GATConv(4 + 32 + 16, 64, heads=2, concat=False),  # 输入: 4(机器人) + 32(环境) + 16(图结构) = 52
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
        self.robot_embedding = nn.Embedding(10, 4)

        self.position_log_std = nn.Parameter(torch.zeros(1, 2))

        # 评估当前状态的价值（Critic） 
        self.critic = nn.Sequential(
            nn.Linear(32 + 16, 64),  # 环境特征 + 平均图特征
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # 状态价值
        )
        
        # # 位置生成分支 - 为每个控制图生成跟随者位置
        # self.position_generators = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(32, 64),
        #         nn.ReLU(),
        #         nn.Linear(64, 2 * self.num_followers)  # 输出极坐标参数
        #     ) for _ in range(num_graphs)
        # ])
        
        # # 编队选择分支
        # self.formation_selector = nn.Sequential(
        #     nn.Linear(32, 16),
        #     nn.ReLU(),
        #     nn.Linear(16, num_graphs)
        # )
        
        # 位置微调模块（用于后期强化学习微调）
        self.position_refiners = nn.ModuleList([
            nn.Sequential(
                nn.Linear(32 + 2 * self.num_followers, 32),
                nn.ReLU(),
                nn.Linear(32, 2 * self.num_followers)
            ) for _ in range(num_graphs)
        ])
        
    def forward(self, env_features, robot_count, candidate_graphs, training_phase="imitation"):
        """
        前向传播
        
        Args:
            env_features: 环境特征 [batch_size, feature_dim]
            training_phase: 训练阶段 ("imitation", "mixed", "rl_finetune")
            robot_count: 整数，机器人数量
            candidate_graphs: 列表，每个元素是一个邻接矩阵 [robot_count, robot_count]
                            不同机器人数量对应的图数量不同
        """
        batch_size = env_features.size(0) # 获取批量大小
        num_graphs = len(candidate_graphs) # 获取控制图数量

        #print("batch_size:", batch_size)
        
        # 特征编码
        env_features = self.encoder(env_features) # [batch_size, 32]

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
        
        # 拼接特征并计算分数
        combined = torch.cat([env_expanded, graph_expanded], dim=-1)  # [batch, num_graphs, 48]

        # 重塑为 [batch * num_graphs, 48] 以便批量处理
        combined_flat = combined.view(-1, 48)
        scores_flat = self.graph_scorer(combined_flat)  # [batch * num_graphs, 1]
        
        # 恢复形状
        graph_scores = scores_flat.view(batch_size, num_graphs)  # [batch, num_graphs]

        # 为每个候选图生成位置分布
        position_dists = []
        
        # 预计算机器人ID嵌入
        robot_emb = self.robot_embedding(robot_ids)  # [batch * robot_count, 4]
        
        # 预计算扩展的环境特征（每个机器人一份）
        env_per_robot = env_features.repeat_interleave(robot_count, dim=0)  # [batch * robot_count, 32]

        # 对每个候选图独立处理
        for i, graph in enumerate(candidate_graphs):
            # 获取当前图的特征
            graph_feat = all_graph_features[i]  # [16]
            graph_feat_per_robot = graph_feat.unsqueeze(0).expand(batch_size * robot_count, -1)  # [batch * robot_count, 16]
            
            # 构建GNN输入
            gnn_input = torch.cat([
                robot_emb,           # [batch * robot_count, 4]
                env_per_robot,       # [batch * robot_count, 32]
                graph_feat_per_robot # [batch * robot_count, 16]
            ], dim=1)  # [batch * robot_count, 52]
            
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
            
            # 生成位置均值
            position_mean = self.position_generator(follower_features)  # [batch, robot_count-1, 2]

            # 创建位置分布
            position_log_std = self.position_log_std.expand_as(position_mean)
            position_dist = torch.distributions.Normal(
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









        # all_results = []
        # all_scores = []
        
        # # 2. 对每个候选控制图单独处理
        # for graph_idx, adj_matrix in enumerate(candidate_graphs):
        #     # adj_matrix: [robot_count, robot_count]

        #     pad_size = (0, 10 - robot_count, 0, 10 - robot_count)  # (左,右,上,下)
        #     adj_matrix_padded = torch.nn.functional.pad(adj_matrix, pad_size, mode='constant', value=0).to(env_features.device)
            
        #     # 2.1 编码图结构
        #     graph_flat = adj_matrix_padded.flatten().unsqueeze(0).repeat(batch_size, 1).to(env_features.device)  # flatten：把 2D 邻接矩阵拉成 1D 向量，unsqueeze：在第 0 维添加一个维度，repeat：复制 batch_size 次
        #     graph_feat = self.graph_encoder(graph_flat)  # [batch_size, 16]
            
        #     # 2.2 构建机器人节点特征
        #     node_features_list = []
        #     for robot_idx in range(robot_count):
        #         # 机器人ID特征
        #         robot_id_feat = self.robot_embedding(
        #             torch.tensor([robot_idx], device=env_features.device)
        #         ).expand(batch_size, -1)  # [batch_size, 4]
                
        #         # 合并：机器人ID + 环境特征 + 图特征
        #         node_feat = torch.cat([
        #             robot_id_feat,  # [batch_size, 4]
        #             env_encoded,    # [batch_size, 32]
        #             graph_feat      # [batch_size, 16]
        #         ], dim=1)  # [batch_size, 52]
                
        #         node_features_list.append(node_feat)
            
        #     node_features = torch.stack(node_features_list, dim=1)  # [batch_size, robot_count, 52]
            
        #     # 2.3 构建图边
        #     edge_index = self._adj_matrix_to_edge_index(adj_matrix).to(env_features.device)
        #     edge_index_batch = self._batch_edge_index(edge_index, robot_count, batch_size)
            
        #     # 2.4 GNN处理（PyG 的 GAT 层只认这种“节点×特征”格式）
        #     node_features_flat = node_features.view(-1, 52)  # [batch_size * robot_count, 52]把三维张量拍成二维，不拷贝数据，只换视图，方便后续层处理。
            
        #     for gnn_layer in self.gnn_layers:
        #         node_features_flat = gnn_layer(node_features_flat, edge_index_batch)
        #         node_features_flat = F.relu(node_features_flat) # ReLU 激活函数
            
        #     # 恢复形状
        #     node_features_out = node_features_flat.view(batch_size, robot_count, -1)  # [batch_size, robot_count, 64]
            
        #     # 2.5 生成位置
        #     # 只生成跟随者位置
        #     follower_polar_params = []  # 存储极坐标参数
        
        #     for follower_idx in range(1, robot_count):
        #         follower_feat = node_features_out[:, follower_idx, :]  # [batch_size, 64]
                
        #         # 生成极坐标参数，然后通过约束函数转换
        #         polar_param = self.position_generator(follower_feat)  # [batch_size, 2]
        #         follower_polar_params.append(polar_param)
            
        #     # 将极坐标参数堆叠
        #     polar_params_tensor = torch.stack(follower_polar_params, dim=1)  # [batch_size, num_followers, 2]
            
        #     # 应用您的极坐标约束转换为直角坐标
        #     follower_positions_tensor = self._polar_to_constrained_cartesian(polar_params_tensor)
            
        #     # 2.6 计算编队分数
        #     # 全局特征（平均池化）
        #     global_feat = torch.mean(node_features_out, dim=1)  # [batch_size, 64]
        #     # 与图特征合并
        #     score_input = torch.cat([global_feat, graph_feat], dim=1)  # [batch_size, 80]
        #     scores = self.formation_scorer(score_input)  # [batch_size, 1]
            
        #     all_results.append(follower_positions_tensor) # [batch_size, robot_count, 2]
        #     all_scores.append(scores) # [batch_size, 1]
        
        # # 3. 合并结果
        # # 注意：不同候选图可能有不同数量，不能直接stack
        # # 我们需要保持列表形式，或者填充到最大数量
        # if len(candidate_graphs) > 0:
        #     # 找到最大机器人数量（通常相同）
        #     max_robots_in_batch = robot_count
            
        #     # 将分数堆叠
        #     scores_tensor = torch.cat(all_scores, dim=1)  # [batch_size, num_candidate_graphs]
            
        #     # 将位置堆叠
        #     follower_positions_stacked = torch.stack(all_results, dim=1)  # [batch_size, num_candidate_graphs, robot_count, 2]
        #     # 添加领航者位置 (0, 0)
        #     leader_positions = torch.zeros(batch_size, len(candidate_graphs), 1, 2, 
        #                                 device=env_features.device)
            
        #     # 完整位置
        #     full_positions = torch.cat([leader_positions, follower_positions_stacked], dim=2)  # [batch_size, num_graphs, robot_count, 2]
        # else:
        #     full_positions = torch.zeros(batch_size, 0, robot_count, 2, device=env_features.device)
        #     scores_tensor = torch.zeros(batch_size, 0, device=env_features.device)






        # # 基础位置生成
        # base_positions = []
        # for i, generator in enumerate(self.position_generators):
        #     # 通过神经网络生成极坐标参数 [batch_size, num_followers, 2]
        #     polar_params = generator(encoded).view(batch_size, self.num_followers, 2)
            
        #     # 应用约束转换为直角坐标
        #     positions = self._polar_to_constrained_cartesian(polar_params) # [batch_size, num_followers, 2]
        #     base_positions.append(positions) # [num_graphs, batch_size, num_followers, 2]

        # # 根据训练阶段决定是否使用微调
        # if training_phase == "rl_finetune":
        #     refined_positions = []
        #     for i, (base_pos, refiner) in enumerate(zip(base_positions, self.position_refiners)):
        #         # 将基础位置和编码特征结合进行微调
        #         # refiner_input是[batch_size, 32 + num_followers*2]
        #         refiner_input = torch.cat([encoded, base_pos.view(batch_size, -1)], dim=1) # base_pos.view(batch_size, -1)是[batch_size, num_followers*2]
        #         delta_polar = refiner(refiner_input).view(batch_size, self.num_followers, 2) # [batch_size, num_followers, 2]
                
        #         # 应用小幅度调整（限制调整幅度）
        #         delta_polar = torch.tanh(delta_polar) * 0.1  # 限制在±0.1范围内
        #         refined_polar = self._cartesian_to_polar(base_pos) + delta_polar
                
        #         # 重新应用约束
        #         refined_pos = self._polar_to_constrained_cartesian(refined_polar) ## 应用约束转换为直角坐标
        #         refined_positions.append(refined_pos)
            
        #     all_positions = refined_positions
        # else:
        #     all_positions = base_positions #[]
        
        # # 编队选择分数
        # # 这个是从神经网络得到的分数，并不是计算的评估分数
        # formation_scores = self.formation_selector(encoded) # 这个只是控制图的分数 [batch_size, num_graphs]
        
        # # 堆叠所有位置配置 
        # positions_tensor = torch.stack(all_positions, dim=1) # [batch_size, num_graphs, num_followers, 2]
        
        # 添加领航者位置 (0, 0)
        # leader_positions = torch.zeros(batch_size, self.num_graphs, 1, 2, device=x.device)
        # full_positions = torch.cat([leader_positions, positions_tensor], dim=2)
        
        # return full_positions, scores_tensor
    
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

class HybridLoss(nn.Module):
    """
    混合损失函数：结合模仿学习和强化学习
    """
    
    def __init__(self, imitation_weight=0.7, rl_weight=0.3, diversity_weight=0.1, control_graphs=None):
        super(HybridLoss, self).__init__()
        self.imitation_weight = imitation_weight
        self.rl_weight = rl_weight
        self.diversity_weight = diversity_weight
        self.control_graphs = control_graphs
        
        self.mse_loss = nn.MSELoss() # 均方误差损失（预测位置与专家位置的差异）连续
        self.ce_loss = nn.CrossEntropyLoss() # 交叉熵损失（预测编队分数与专家选择的差异）离散
    
    def forward(self, pred_positions, pred_scores, expert_positions, 
                expert_graph, advantages, has_expert_mask):
        """
        计算混合损失
        
        Args:
            pred_positions: 预测的位置 [batch_size, num_graphs, num_robots, 2]
            pred_scores: 预测的编队分数 [batch_size, num_graphs]
            expert_positions: 专家位置 [batch_size, num_robots, 2]
            expert_graph: 专家选择的编队控制图 shape [batch_size, num_robots, num_robots]
            control_graphs: 控制图列表 shape (num_graphs, num_robots, num_robots)
            expert_graph_idx: 专家选择的编队索引 [batch_size]
            advantages: 优势函数 [batch_size, num_graphs]
            has_expert_mask: 是否有专家标注 [batch_size]
        """
        batch_size = pred_positions.size(0)

        # 获取专家选择的控制图索引
        expert_graph_idx = torch.full((batch_size,), -1, dtype=torch.long, device=pred_positions.device)
        for i in range(batch_size):
            for j in range(self.control_graphs.shape[0]):
                if torch.equal(expert_graph[i], self.control_graphs[j]):
                    expert_graph_idx[i] = j
                    break

        # print("control_graphs:", self.control_graphs)  
        # print("expert_graph:", expert_graph)
        # print("expert_graph_idx:", expert_graph_idx)
        
        
        # 1. 模仿学习损失（仅对有专家标注的样本）
        imitation_loss = torch.tensor(0.0, device=pred_positions.device, requires_grad=False)
        if has_expert_mask.any():
            # 位置损失
            pred_expert_positions = pred_positions[has_expert_mask, expert_graph_idx[has_expert_mask]]

            # print("pred_expert_positions:", pred_expert_positions)
            # print("expert_positions:", expert_positions[has_expert_mask])
            # print("has_expert_mask:", has_expert_mask)
            # print("expert_positions:", expert_positions)
            # # print("has_expert_mask 数据类型：", has_expert_mask.dtype)
            # sys.exit("stop here")
            position_loss = self.mse_loss(pred_expert_positions, expert_positions[has_expert_mask])
            
            # 分类损失（控制图选择）
            classification_loss = self.ce_loss(pred_scores[has_expert_mask], expert_graph_idx[has_expert_mask])
            # print("pred_scores:", pred_scores[has_expert_mask])
            # print("expert_graph_idx:", expert_graph_idx[has_expert_mask])
            

            # print(position_loss, classification_loss)
            # sys.exit("stop here")

            imitation_loss = position_loss + classification_loss
        
        # 2. 强化学习损失（对所有样本）但是这个pred_scores指的是控制图的分数，不是位置
        # 目前由于位置是连续的，所以强化学习损失只作用于控制图选择上，之后可以尝试一下用其他强化学习方法来优化位置（比如PPO）
        # print("advantages:", advantages)
        # print("pred_scores:", pred_scores)
        # sys.exit("stop here")
        log_probs = F.log_softmax(pred_scores, dim=1) # 获取对数概率
        rl_loss = -torch.mean(log_probs * advantages) # advantages代表了奖励
        
        # 3. 多样性损失（鼓励不同控制图生成不同的位置）
        diversity_loss = self._compute_diversity_loss(pred_positions)

        # self.rl_weight = 0.0
        # self.diversity_weight = 0.0
        
        # 加权组合
        total_loss = (self.imitation_weight * imitation_loss + 
                     self.rl_weight * rl_loss + 
                     self.diversity_weight * diversity_loss)
        
        # print("total_loss, imitation_loss, rl_loss, diversity_loss", total_loss, imitation_loss, rl_loss, diversity_loss)
        
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