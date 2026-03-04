import numpy as np
import torch
import torch.nn.functional as F
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.torch_utils import apply_grad_clipping
from ray.rllib.evaluation import compute_advantages
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper

torch, nn = try_import_torch()

class FormationActionDistribution(TorchDistributionWrapper):
    """自定义混合动作分布：离散图选择 + 连续位置（继承了TorchDistributionWrapper）"""
    
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
    
    def deterministic_sample(self):
        """返回确定性动作（用于评估模式）"""
        graph_idx = torch.argmax(self.graph_logits, dim=-1)
        positions = self.position_dist.mean
        positions_flat = positions.view(positions.shape[0], -1)
        action = torch.cat([
            graph_idx.unsqueeze(1).float(),
            positions_flat
        ], dim=1)
        return action
    
    def logp(self, actions):
        """计算动作的对数概率"""
        graph_idx = actions[:, 0].long()
        positions = actions[:, 1:].view(-1, self.model.num_followers, 2)
        graph_logp = self.graph_dist.log_prob(graph_idx)
        position_logp = self.position_dist.log_prob(positions).sum(dim=[1, 2])
        return graph_logp + position_logp
    
    def entropy(self):
        """计算分布熵"""
        return self.graph_dist.entropy() + self.position_dist.entropy().mean(dim=[1, 2])
    
    def kl(self, other=None):
        """KL散度（简化实现，返回0）"""
        return torch.tensor(0.0, device=self.graph_logits.device)
    
    def sampled_action_logp(self):
        """返回最后一次采样动作的对数概率"""
        return self.logp(self.sample())
    


def custom_action_distribution_fn(model, obs_batch, **kwargs):
    # 类似地，处理 obs_batch
    if hasattr(obs_batch, "keys") and "obs" in obs_batch:
        obs = obs_batch["obs"]
    elif isinstance(obs_batch, torch.Tensor):
        obs = obs_batch
    else:
        obs = torch.from_numpy(np.array(obs_batch)).float()

    # 移动到与模型相同的设备
    device = next(model.parameters()).device
    obs = obs.to(device)

    input_dict = {"obs": obs}
    logits, state = model(input_dict, [], None)
    return logits, FormationActionDistribution, state

def custom_extra_action_out_fn(policy, input_dict, state_batches, model, action_dist):
    extra_outputs = {}
    if hasattr(model, '_last_model_output'):
        value = model._last_model_output.get('value')
        if value is not None:
            if torch.is_tensor(value):
                value = value.detach().cpu().numpy()
            if value.ndim == 2 and value.shape[1] == 1:
                value = value.flatten()  # 展平为一维
            extra_outputs["value"] = value
    return extra_outputs

def custom_loss_fn(policy, model, dist_class, train_batch):
    # 获取数据，如果缺失则使用虚拟数据（初始化阶段）
    obs = train_batch.get(SampleBatch.OBS, None)
    actions = train_batch.get(SampleBatch.ACTIONS, None)
    old_action_logp = train_batch.get(SampleBatch.ACTION_LOGP, None)
    advantages = train_batch.get("advantages", None)
    returns = train_batch.get("returns", None)
    old_values = train_batch.get(SampleBatch.VF_PREDS, None)

    # print("\n=== Train Batch Field Check ===")
    # fields = {
    #     "obs": obs,
    #     "actions": actions,
    #     "old_action_logp": old_action_logp,
    #     "advantages": advantages,
    #     "returns": returns,
    #     "old_values": old_values
    # }
    
    # for name, value in fields.items():
    #     if value is None:
    #         print(f"{name}: None ❌")
    #     else:
    #         # 打印形状+数据类型+是否全0（核心验证）
    #         shape = value.shape if hasattr(value, 'shape') else "no shape"
    #         dtype = value.dtype if hasattr(value, 'dtype') else type(value)
    #         is_all_zero = False
    #         if isinstance(value, (np.ndarray, torch.Tensor)):
    #             is_all_zero = np.allclose(value, 0) if isinstance(value, np.ndarray) else torch.allclose(value, torch.tensor(0.0))
            
    #         print(f"{name}: shape={shape}, dtype={dtype}, all_zero={is_all_zero} ✅")
    #         # 额外打印前3个值（直观看数据是否有效）
    #         if isinstance(value, (np.ndarray, torch.Tensor)) and len(value) > 0:
    #             print(f"  → first 3 values: {value[:3]}")
    # print(f"advantages raw: {advantages}")  # 如果是 numpy 数组，打印前几个值

    # print(f"advantages mean: {advantages.mean()}, advantages std: {advantages.std()}")
    # print(f"returns mean: {returns.mean()}, returns std: {returns.std()}")


    # 如果任何关键数据缺失，使用虚拟数据（发生在 dummy batch 初始化时）
    if any(v is None for v in [obs, actions, old_action_logp, advantages, returns, old_values]):
        print("Warning: Some train batch fields are None. Using dummy data.")
        obs_shape = policy.observation_space.shape
        action_shape = policy.action_space.shape
        obs = torch.zeros((1, *obs_shape), device=policy.device)
        actions = torch.zeros((1, *action_shape), device=policy.device)
        old_action_logp = torch.zeros(1, device=policy.device)
        advantages = torch.zeros(1, device=policy.device)
        returns = torch.zeros(1, device=policy.device)
        old_values = torch.zeros(1, device=policy.device)
    else:
        # 转换 numpy 为张量
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float().to(policy.device)
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).float().to(policy.device)
        if isinstance(old_action_logp, np.ndarray):
            old_action_logp = torch.from_numpy(old_action_logp).float().to(policy.device)
        if isinstance(advantages, np.ndarray):
            advantages = torch.from_numpy(advantages).float().to(policy.device)
            print(f"advantages raw: {advantages}") 
        if isinstance(returns, np.ndarray):
            returns = torch.from_numpy(returns).float().to(policy.device)
        if isinstance(old_values, np.ndarray):
            old_values = torch.from_numpy(old_values).float().to(policy.device)

    # 前向传播
    input_dict = {SampleBatch.OBS: obs}
    logits, _ = model(input_dict, [], None)
    dist = FormationActionDistribution(logits, model)

    new_action_logp = dist.logp(actions)
    new_values = model.value_function()

    # 确保形状匹配（假设 value_function 返回 [batch] 或 [batch,1]）
    if new_values.dim() == 2:
        new_values = new_values.squeeze(-1)
    if returns.dim() == 2:
        returns = returns.squeeze(-1)
    if old_values.dim() == 2:
        old_values = old_values.squeeze(-1)

    # PPO clip损失
    ratio = torch.exp(new_action_logp - old_action_logp)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - policy.config["clip_param"], 1.0 + policy.config["clip_param"]) * advantages
    action_loss = -torch.min(surr1, surr2).mean()

    # 价值损失
    value_loss = 0.5 * (new_values - returns).pow(2).mean()

    # 熵正则化
    entropy = dist.entropy().mean()
    entropy_loss = -policy.config["entropy_coeff"] * entropy

    total_loss = action_loss + policy.config["vf_loss_coeff"] * value_loss + entropy_loss

    # 构建统计字典
    stats = {
        "total_loss": total_loss.detach().cpu().item(),
        "policy_loss": action_loss.detach().cpu().item(),
        "vf_loss": value_loss.detach().cpu().item(),
        "entropy": entropy.detach().cpu().item(),
        "kl": 0.0,  # 占位符
        "vf_explained_var": 0.0,
    }

    # print(f"train_batch keys: {train_batch.keys()}")
    # if SampleBatch.VF_PREDS in train_batch:
    #     print(f"vf_preds shape: {train_batch[SampleBatch.VF_PREDS].shape}, values: {train_batch[SampleBatch.VF_PREDS][:5]}")

    # print(f"value mean: {new_values.mean().item()}, value std: {new_values.std().item()}")
    # print(f"returns mean: {returns.mean().item()}, advantages mean: {advantages.mean().item()}")

    return total_loss, stats

def custom_postprocess_fn(policy, sample_batch, other_agent_batches, episode):


    obs = sample_batch[SampleBatch.OBS]
    # print(f"Obs sample: {obs[:2]}") 

    # 确保有价值预测
    if SampleBatch.VF_PREDS not in sample_batch:
        if "value" in sample_batch:
            sample_batch[SampleBatch.VF_PREDS] = sample_batch["value"]
        else:
            # 手动计算价值（仅用于调试）
            obs = sample_batch[SampleBatch.OBS]
            values = []
            for o in obs:
                input_dict = {"obs": torch.from_numpy(o).float().to(policy.device).unsqueeze(0)}
                with torch.no_grad():
                    policy.model(input_dict, [], None)
                    v = policy.model.value_function().item()
                values.append(v)
            sample_batch[SampleBatch.VF_PREDS] = np.array(values, dtype=np.float32)

    # 将 vf_preds 展平为一维（如果它是二维的）
    vf_preds = sample_batch[SampleBatch.VF_PREDS]
    if vf_preds.ndim == 2 and vf_preds.shape[1] == 1:
        vf_preds = vf_preds.flatten()
        sample_batch[SampleBatch.VF_PREDS] = vf_preds

    # 计算最后一步的价值
    completed = sample_batch[SampleBatch.DONES][-1]
    if completed:
        last_r = 0.0
    else:
        last_obs_np = sample_batch[SampleBatch.NEXT_OBS][-1]
        last_obs = torch.from_numpy(last_obs_np).float().to(policy.device).unsqueeze(0)
        with torch.no_grad():
            input_dict = {"obs": last_obs}
            policy.model(input_dict, [], None)
            last_r = policy.model.value_function().item()  # 已经是标量
            
    rewards = sample_batch[SampleBatch.REWARDS]
    vf_preds = sample_batch[SampleBatch.VF_PREDS]
    # print(f"Rewards[:5]: {rewards[:5]}, VF preds[:5]: {vf_preds[:5]}")

    # 计算优势
    post_batch = compute_advantages(
        sample_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"],
        use_critic=policy.config.get("use_critic", True)
    )

    vf_preds = sample_batch[SampleBatch.VF_PREDS]
    advantages = post_batch["advantages"]
    # 确保形状一致
    if len(advantages.shape) > 1:
        advantages = advantages.flatten()
    if len(vf_preds.shape) > 1:
        vf_preds = vf_preds.flatten()
    # 计算returns（这是PPO的标准公式）
    returns = advantages + vf_preds
    # 把returns写入post_batch
    post_batch["returns"] = returns

    #print(f"Computed advantages[:5]！！！！！！！！！！！！！！！: {post_batch['advantages'][:5]}")

    # 确保返回的批次包含 advantages 和 returns
    # if "advantages" not in post_batch or "returns" not in post_batch:
    #     #print("WARNING: compute_advantages did not add advantages/returns")
    #     post_batch["advantages"] = np.zeros_like(post_batch[SampleBatch.REWARDS])
    #     post_batch["returns"] = post_batch[SampleBatch.REWARDS].copy()

    # print(f"\n=== postprocess_fn返回前 FINAL CHECK ===")
    # print(f"post_batch是否有advantages: {'advantages' in post_batch}")
    # print(f"post_batch advantages[:5]: {post_batch['advantages'][:5]}")
    # print(f"post_batch类型: {type(post_batch)}")  # 必须是SampleBatch/MultiAgentBatch

    return post_batch

class CustomFormationPolicy(PPOTorchPolicy):
    """自定义PPO策略，使用你定义的函数"""

    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        self._last_stats = {}
    
    def loss(self, model, dist_class, train_batch):
        total_loss, stats = custom_loss_fn(self, model, dist_class, train_batch)
        self._last_stats = stats
        return total_loss

    def postprocess_trajectory(self, sample_batch, other_agent_batches=None, episode=None):
        return custom_postprocess_fn(self, sample_batch, other_agent_batches, episode)

    def extra_action_out(self, input_dict, state_batches, model, action_dist):
        return custom_extra_action_out_fn(self, input_dict, state_batches, model, action_dist)

    def action_distribution_fn(self, model, obs_batch, **kwargs):
        return custom_action_distribution_fn(model, obs_batch, **kwargs)
    
    def stats_fn(self, train_batch):
        return self._last_stats