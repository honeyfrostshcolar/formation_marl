import numpy as np
import torch
import torch.nn.functional as F
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.torch_utils import apply_grad_clipping
from ray.rllib.evaluation import compute_advantages
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()

class FormationActionDistribution(ActionDistribution):
    """自定义混合动作分布：离散图选择 + 连续位置（继承了ActionDistribution）"""
    
    def __init__(self, inputs, model):
        super().__init__(inputs, model)
        batch_size = inputs.shape[0]
        num_graphs = model.num_graphs
        num_followers = model.num_followers
        
        # 分割输入：图分数 + 位置参数
        self.graph_logits = inputs[:, :num_graphs]  # [batch, num_graphs]
        self.position_params = inputs[:, num_graphs:]  # [batch, 2*num_followers]
        self.position_params = self.position_params.view(batch_size, num_followers, 2)
        
        # 创建分布
        self.graph_dist = torch.distributions.Categorical(logits=self.graph_logits)
        
        # 位置分布：使用正态分布
        position_means = self.position_params
        position_log_std = model.model.position_log_std.expand_as(position_means)
        self.position_dist = torch.distributions.Normal(
            position_means, 
            torch.exp(position_log_std)
        )
    
    def sample(self):
        """采样动作（同时采样离散图索引 + 连续位置，再拼接成环境能识别的动作）"""
        graph_idx = self.graph_dist.sample()
        positions = self.position_dist.sample()
        
        # 合并动作
        positions_flat = positions.view(positions.shape[0], -1)
        action = torch.cat([
            graph_idx.unsqueeze(1).float(),  # 转换为float
            positions_flat
        ], dim=1)
        
        return action
    
    def logp(self, actions):
        """计算动作的对数概率（同时计算离散选择 + 连续位置的对数概率）"""
        # 分割动作
        graph_idx = actions[:, 0].long()
        positions = actions[:, 1:].view(-1, self.model.num_followers, 2)
        
        # 图选择的log prob
        graph_logp = self.graph_dist.log_prob(graph_idx)
        
        # 位置的log prob
        position_logp = self.position_dist.log_prob(positions).sum(dim=[1, 2])
        
        return graph_logp + position_logp
    
    def entropy(self):
        """计算熵（计算混合分布的熵）"""
        return self.graph_dist.entropy() + self.position_dist.entropy().mean(dim=[1, 2])

def custom_action_distribution_fn(policy, model, input_dict, state_batches, explore, timestep):
    """自定义动作分布函数"""
    # 获取模型输出
    logits, state = model(input_dict, state_batches, None)
    
    # 创建自定义分布
    dist = FormationActionDistribution(logits, model)
    
    return dist, state

def custom_extra_action_out_fn(policy, input_dict, state_batches, model, action_dist):
    """额外的动作输出，用于保存采样信息"""
    batch_size = input_dict["obs"].shape[0]
    
    # 从模型中获取完整的输出
    model_output = model._last_model_output
    
    # 准备额外的输出
    extra_outputs = {}
    if hasattr(model, '_last_model_output'):
        extra_outputs["graph_scores"] = model_output['graph_scores']
        extra_outputs["position_dists"] = model_output['position_dists']
        extra_outputs["value"] = model_output['value']
    
    return extra_outputs

def custom_loss_fn(policy, model, dist_class, train_batch):
    """重写损失函数"""
    # 获取训练数据
    obs = train_batch[SampleBatch.OBS]
    actions = train_batch[SampleBatch.ACTIONS]
    rewards = train_batch[SampleBatch.REWARDS]
    dones = train_batch[SampleBatch.DONES]
    old_values = train_batch[SampleBatch.VF_PREDS]
    old_action_logp = train_batch[SampleBatch.ACTION_LOGP]
    
    # 重新前向传播获取新策略输出
    input_dict = {SampleBatch.OBS: torch.from_numpy(obs).float().to(policy.device)}
    logits, _ = model(input_dict, [], None)
    dist = FormationActionDistribution(logits, model)
    
    # 计算新策略的动作概率和价值
    new_action_logp = dist.logp(torch.from_numpy(actions).float().to(policy.device))
    new_values = model.value_function()
    
    # 计算优势
    advantages = torch.from_numpy(train_batch[SampleBatch.ADVANTAGES]).float().to(policy.device)
    returns = torch.from_numpy(train_batch[SampleBatch.RETURNS]).float().to(policy.device)
    
    # PPO clip损失
    ratio = torch.exp(new_action_logp - torch.from_numpy(old_action_logp).float().to(policy.device))
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - policy.config["clip_param"], 1.0 + policy.config["clip_param"]) * advantages
    action_loss = -torch.min(surr1, surr2).mean()
    
    # 价值损失
    value_loss = 0.5 * torch.pow(new_values - returns, 2).mean()
    
    # 熵正则化
    entropy_loss = -policy.config["entropy_coeff"] * dist.entropy().mean()
    
    # 总损失
    total_loss = action_loss + policy.config["vf_loss_coeff"] * value_loss + entropy_loss
    
    # 记录指标
    policy.metrics.info["ppo/ratio"] = ratio.mean().item()
    policy.metrics.info["ppo/action_loss"] = action_loss.item()
    policy.metrics.info["ppo/value_loss"] = value_loss.item()
    policy.metrics.info["ppo/entropy_loss"] = entropy_loss.item()
    
    return total_loss

def custom_postprocess_fn(policy, sample_batch, other_agent_batches, episode):
    """后处理函数，用于计算优势函数"""
    # 使用PPO默认的后处理函数计算GAE
    completed = sample_batch[SampleBatch.DONES][-1]
    if completed:
        last_r = 0.0
    else:
        next_state = []
        last_r = policy._value(sample_batch[SampleBatch.NEXT_OBS][-1], sample_batch[SampleBatch.ACTIONS][-1], 
                              sample_batch[SampleBatch.REWARDS][-1], *next_state)
    
    return compute_advantages(
        sample_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"],
        use_critic=policy.config.get("use_critic", True))