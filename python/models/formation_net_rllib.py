from ray.rllib.models import ModelV2
from ray.rllib.utils.annotations import override
import torch
import torch.nn as nn
from models.formation_net import ConstrainedFormationNet

class ConstrainedFormationNetRLLib(ModelV2):
    """适配RLlib的ConstrainedFormationNet模型"""
    
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        
        custom_config = model_config["custom_model_config"]  # 提取你配置的参数字典
        feature_dim = custom_config["feature_dim"]
        num_graphs = custom_config["num_graphs"]
        max_robots = custom_config["max_robots"]
        min_distance = custom_config.get("min_distance", 0.5)  # 带默认值，防止没传
        max_distance = custom_config.get("max_distance", 3.0)
        control_graphs_np = custom_config["control_graphs"]

        self.num_robots = model_config.get("num_robots", 3)
        self.num_followers = self.num_robots - 1
        self.num_graphs = num_graphs

        # 初始化模型
        self.model = ConstrainedFormationNet(
            feature_dim=feature_dim,
            num_graphs=self.num_graphs,
            max_robots=max_robots,
            min_distance=min_distance,
            max_distance=max_distance
        )
        
        # 为每个机器人创建候选图
        self.candidate_graphs = [torch.from_numpy(cg).float() for cg in control_graphs_np]  # 从numpy数组转换为torch张量

        # 输出维度：graph_scores + 每个跟随者的位置
        self._num_outputs = num_graphs + 2 * self.num_followers # 图分数 + 每个跟随者的(x, y)(在哪里用到了？)
    
    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        """
            前向传播，返回动作和价值
        Args:
            input_dict: 包含观测数据的字典
            state: 历史状态
            seq_lens: 
        """
        # 获取环境特征
        features = input_dict["obs"]
        
        batch_size = features.shape[0]
        
        # 创建机器人ID
        robot_ids = torch.arange(0, self.num_robots, device=features.device)
        robot_ids = robot_ids.unsqueeze(0).expand(batch_size, -1).reshape(-1)
        
        # 前向传播
        outputs = self.model(
            features=features,
            robot_ids=robot_ids,
            candidate_graphs=self.candidate_graphs,
            robot_count=self.num_robots
        )

        self._last_model_output = outputs  # 保存最后的模型输出，供value_function使用
        
        # 构建logits：图分数 + 位置参数
        graph_scores = outputs['graph_scores']  # [batch, num_graphs]
        
        # 获取第一个图的位置均值作为初始位置参数（其他图的位置均值也可以作为初始参数）（为什么叫初始？每次不都是0吗？）
        position_means = outputs['position_dists'][0].mean  # [batch, num_followers, 2]
        position_means_flat = position_means.view(batch_size, -1)  # [batch, 2*num_followers]
        
        # 合并logits
        logits = torch.cat([graph_scores, position_means_flat], dim=1)
        
        return logits, state
    
    @override(ModelV2)
    def value_function(self):
        """返回价值函数"""
        # 这里返回模型的价值网络输出
        # 实际实现中需要从模型中获取
        return self.model.value