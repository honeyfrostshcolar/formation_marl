import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class ConstrainedFormationNet(nn.Module):
    """
    基于 GAT (多头图注意力网络) 的核心特征提取器。
    具备处理可变机器人数量的潜力 (权重在各节点间共享)。
    """
    def __init__(self, feature_dim, num_graphs, num_robots, max_robots=10):
        super().__init__()
        self.num_graphs = num_graphs
        self.num_robots = num_robots
        self.num_followers = num_robots - 1

        self.max_robots = max_robots
        self.max_followers = max_robots - 1
        
        # 1. 环境特征编码 (雷达数据)
        self.env_encoder = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 64),
            nn.ReLU()
        )
        
        # 2. 机器人身份嵌入 (最多支持 10 个机器人，提取 16 维专属特征)
        # 领航者是 ID 0，跟随者是 ID 1, 2, ...
        self.robot_embedding = nn.Embedding(10, 16)
        
        # 3. GAT 图神经网络层
        # 节点输入维度 = 环境特征(64) + 身份嵌入(16) = 80
        self.gat1 = GATConv(80, 64, heads=2, concat=False)
        self.gat2 = GATConv(64, 64, heads=2, concat=False)
        
        # 4. 备选图结构特征提取 (为了兼容不同数量，统一 padding 到 10x10)
        self.graph_encoder = nn.Sequential(
            nn.Linear(10 * 10, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        
        # 5. 特征融合与隐藏状态提取
        # 拼接跟随者的 GNN 特征 (num_followers * 64) 和 图特征 (16)
        # 压缩到 128 维，对接 RLlib 包装器
        self.hidden_extractor = nn.Sequential(
            nn.Linear(self.max_followers * 64 + 16, 128),  # 永远是 9 * 64 + 16
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # 6. 价值网络 (Critic)
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class ConstrainedFormationNet(nn.Module):
    def __init__(self, feature_dim, num_graphs, max_robots=10):
        super().__init__()
        self.num_graphs = num_graphs
        
        # ✅ 网络永远以 max_robots 为基准建立
        self.max_robots = max_robots
        self.max_followers = max_robots - 1
        
        self.env_encoder = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 64),
            nn.ReLU()
        )
        
        # 预留 10 个 ID 位置
        self.robot_embedding = nn.Embedding(self.max_robots, 16)
        
        self.gat1 = GATConv(80, 64, heads=2, concat=False)
        self.gat2 = GATConv(64, 64, heads=2, concat=False)
        
        self.graph_encoder = nn.Sequential(
            nn.Linear(10 * 10, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        
        # ✅ 提取层永远接收 9 个跟随者的特征
        self.hidden_extractor = nn.Sequential(
            nn.Linear(self.max_followers * 64 + 16, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    # ✅ 前向传播接收当前实际的 current_num_robots
    def forward(self, features, candidate_graphs, current_num_robots):
        batch_size = features.size(0)
        device = features.device
        
        # 1. 备选图特征提取
        graph_features_list = []
        for graph in candidate_graphs:
            pad_size = (0, 10 - current_num_robots, 0, 10 - current_num_robots)
            graph_padded = F.pad(graph, pad_size, mode='constant', value=0)
            graph_feat = self.graph_encoder(graph_padded.flatten().unsqueeze(0))
            graph_features_list.append(graph_feat)
            
        all_graph_features = torch.cat(graph_features_list, dim=0)
        mean_graph_feat = torch.mean(all_graph_features, dim=0, keepdim=True).expand(batch_size, -1)
        
        # 2. 准备 GNN 节点特征 (永远生成 max_robots 个节点)
        env_feat = self.env_encoder(features) 
        env_feat_expanded = env_feat.unsqueeze(1).expand(-1, self.max_robots, -1)
        
        robot_ids = torch.arange(self.max_robots, device=device)
        robot_embs = self.robot_embedding(robot_ids).unsqueeze(0).expand(batch_size, -1, -1)
        
        node_features = torch.cat([env_feat_expanded, robot_embs], dim=-1)
        node_features_flat = node_features.view(batch_size * self.max_robots, -1)
        
        # 3. ✅ 核心掩码逻辑：只给前 current_num_robots 个真实节点连线！
        # 后面多出来的假节点，没有任何边相连，它们只会被孤立在旁边，无法干扰真实节点
        edge_list = []
        for b in range(batch_size):
            offset = b * self.max_robots
            for i in range(current_num_robots):
                for j in range(current_num_robots):
                    edge_list.append([offset + i, offset + j])
                    
        edge_index = torch.tensor(edge_list, dtype=torch.long, device=device).t().contiguous()
        
        # 4. GNN 前向传播
        x = self.gat1(node_features_flat, edge_index)
        x = F.relu(x)
        x = self.gat2(x, edge_index)
        x = F.relu(x)
        
        x_reshaped = x.view(batch_size, self.max_robots, -1)
        
        # 5. 提取所有跟随者特征 (固定 9 个)
        follower_features = x_reshaped[:, 1:, :] # 取 1 到 9
        follower_flat = follower_features.reshape(batch_size, -1)
        
        combined_feat = torch.cat([follower_flat, mean_graph_feat], dim=-1)
        hidden_state = self.hidden_extractor(combined_feat)
        value = self.critic(hidden_state).squeeze(-1)
        
        return hidden_state, value