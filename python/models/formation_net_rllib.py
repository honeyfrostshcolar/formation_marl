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

        # 初始化模型
        self.model = ConstrainedFormationNet(
            feature_dim=feature_dim,
            num_graphs=num_graphs,
            max_robots=max_robots,
            min_distance=min_distance,
            max_distance=max_distance
        )
        
        # 为每个机器人创建候选图
        self.candidate_graphs = [torch.from_numpy(cg).float() for cg in control_graphs_np]  # 从numpy数组转换为torch张量
    
    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        """前向传播，返回动作和价值"""
        # 获取环境特征
        features = input_dict["obs"]
        
        # 机器人数量（固定为3，实际应用中可能需要动态获取）
        robot_count = 3
        
        # 前向传播
        outputs = self.model(
            features=features,
            robot_ids=torch.arange(0, robot_count),
            candidate_graphs=self.candidate_graphs,
            robot_count=robot_count
        )
        
        # 提取输出
        graph_scores = outputs['graph_scores']  # [batch_size, num_graphs]
        position_dists = outputs['position_dists']  # 列表，长度num_graphs
        
        # 返回
        return graph_scores, state, position_dists
    
    @override(ModelV2)
    def value_function(self):
        """返回价值函数"""
        # 这里返回模型的价值网络输出
        # 实际实现中需要从模型中获取
        return self.model.value