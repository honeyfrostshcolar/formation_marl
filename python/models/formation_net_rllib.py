import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from models.formation_net import ConstrainedFormationNet

class ConstrainedFormationNetRLLib(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        custom_config = model_config.get("custom_model_config", {})
        self.feature_dim = custom_config.get("feature_dim", 21)
        
        self.candidate_graphs = [torch.from_numpy(cg).float() for cg in custom_config["control_graphs"]]
        self.current_num_robots = custom_config.get("num_robots", 3)
        self.max_robots = 10
        
        # 1. 初始化核心网络：这里不传 num_robots，因为网络永远按 10 个节点的最大容量构建
        self.core_network = ConstrainedFormationNet(
            feature_dim=self.feature_dim,
            num_graphs=len(self.candidate_graphs),
            max_robots=self.max_robots
        )
        
        # RLlib 自动算好了混合动作空间需要的 logit 维度
        self.action_head = nn.Linear(128, num_outputs)
        
        self._cur_value = None

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"].float()
        device = next(self.core_network.parameters()).device
        obs = obs.to(device)
        candidate_graphs = [g.to(device) for g in self.candidate_graphs]
        
        # 2. 前向传播：必须调用实例化的 self.core_network！
        # 并且在这里把【当前的真实机器人数量】(current_num_robots) 传进去，告诉 GNN 连几根线
        hidden_state, self._cur_value = self.core_network(
            features=obs, 
            candidate_graphs=candidate_graphs, 
            current_num_robots=self.current_num_robots
        )
        
        logits = self.action_head(hidden_state)
        
        return logits, state

    def value_function(self):
        return self._cur_value