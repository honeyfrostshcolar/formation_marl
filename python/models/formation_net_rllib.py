from ray.rllib.models import ModelV2
from ray.rllib.utils.annotations import override
import torch
import torch.nn as nn
import sys
from models.formation_net import ConstrainedFormationNet


class ConstrainedFormationNetRLLib(ModelV2):
    """适配RLlib的ConstrainedFormationNet模型"""
    
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        framework = kwargs.get("framework", "torch")
        super().__init__(obs_space, action_space, num_outputs, model_config, name, framework)
        
        custom_config = model_config["custom_model_config"]  # 提取你配置的参数字典
        feature_dim = custom_config["feature_dim"]
        num_graphs = custom_config["num_graphs"]
        max_robots = custom_config["max_robots"]
        min_distance = custom_config.get("min_distance", 0.5)  # 带默认值，防止没传
        max_distance = custom_config.get("max_distance", 3.0)
        control_graphs_np = custom_config["control_graphs"]

        self.num_robots = custom_config.get("num_robots", 3) 
        self.num_followers = self.num_robots - 1
        self.num_graphs = num_graphs

        # 初始化模型
        self.model = ConstrainedFormationNet(
            feature_dim=feature_dim,
            num_graphs=self.num_graphs,
            num_robots=self.num_robots,
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
            前向传播，返回动作和价值（模型 forward 方法接收到的是 SampleBatch 对象，而不是预期的张量）
        Args:
            input_dict: 包含观测数据的字典
            state: 历史状态
            seq_lens: 
        """

        # 获取环境特征
        raw_obs = input_dict["obs"]
    
        # 判断 raw_obs 的类型，提取观测张量
        if hasattr(raw_obs, "keys") and "obs" in raw_obs:
            # 如果是 SampleBatch 或字典，通过键 "obs" 获取
            features = raw_obs["obs"]
        elif isinstance(raw_obs, torch.Tensor):
            # 已经是张量，直接使用
            features = raw_obs
        else:
            # 其他情况（如 numpy 数组），转换为张量
            features = torch.from_numpy(np.array(raw_obs)).float()

        # 确保 features 是 torch.Tensor 且在正确的设备上
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32)
        
        # 移动到与模型相同的设备
        device = next(self.model.parameters()).device
        features = features.to(device)
        
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
        
        # 获取第一个图的位置均值作为连续动作的参数[0]表示位置均值，[1]表示位置标准差
        position_means = outputs['position_dists'][0].mean  # [batch, num_followers, 2]
        position_means_flat = position_means.view(batch_size, -1)  # [batch, 2*num_followers]
        
        # 合并logits
        logits = torch.cat([graph_scores, position_means_flat], dim=1)
        
        return logits, state
    
    @override(ModelV2)
    def value_function(self):
        """返回价值函数"""
        # 从保存的最后一次模型输出中获取价值
        if hasattr(self, '_last_model_output'):
            value = self._last_model_output.get('value')
            if value is not None:
                # 确保返回形状为 (batch_size,) 或 (batch_size, 1)
                return value.squeeze(-1)  # 假设 shape 为 [batch, 1]
        # 如果还没有输出，返回一个零张量
        return torch.zeros(1, device=next(self.model.parameters()).device)
    
    def parameters(self, recurse=True):
        """返回内部模型的可训练参数"""
        return self.model.parameters(recurse=recurse)
    
    def train(self, mode=True):
        """将内部模型设置为训练模式"""
        self.model.train(mode)
        return self

    def eval(self):
        """将内部模型设置为评估模式"""
        self.model.eval()
        return self

    def to(self, *args, **kwargs):
        """将内部模型移动到指定设备"""
        self.model = self.model.to(*args, **kwargs)
        return self
    
    def state_dict(self, *args, **kwargs):
        """返回内部模型的状态字典"""
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        """加载状态字典到内部模型"""
        return self.model.load_state_dict(*args, **kwargs)
    
    def named_parameters(self, *args, **kwargs):
        return self.model.named_parameters(*args, **kwargs)

    def named_buffers(self, *args, **kwargs):
        return self.model.named_buffers(*args, **kwargs)