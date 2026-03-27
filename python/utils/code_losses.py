"""损失函数定义。

这个文件把论文中用到的损失全部拆开写清楚：
- L_inf: future action inference loss
- L_c: continuity loss
- L_k: KL loss
- L_int: 意图学习总损失
- L_e: 在 fusion 模块里已经算出
- L_total: 总损失 = L_RL + L_int + L_e

注意：
- `L_e` 依赖注意力权重 alpha，所以它在 fusion 模块里计算更自然。
- 这里负责把其他损失算出来，并与外部 RL loss 合并。
"""

from typing import Dict

import torch
import torch.nn.functional as F

from .config import CoDeConfig


def inference_loss(
    pred_actions: torch.Tensor,
    target_actions: torch.Tensor,
    discrete_action: bool,
) -> torch.Tensor:
    """计算论文里的 inference loss，记为 L_inf。

    含义：
    让当前 intent 真的能预测未来动作序列。

    输入：
    - 离散动作时：
      - pred_actions: [B, K, A]，每步动作 logits
      - target_actions: [B, K]，每步动作 id
    - 连续动作时：
      - pred_actions: [B, K, A]
      - target_actions: [B, K, A]
    """
    if discrete_action:
        batch_size, horizon, action_dim = pred_actions.shape
        return F.cross_entropy(
            pred_actions.reshape(batch_size * horizon, action_dim),
            target_actions.reshape(batch_size * horizon).long(),
        )

    return F.mse_loss(pred_actions, target_actions)


def continuity_loss(
    intent_prev: torch.Tensor,
    intent_now: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """计算论文里的 continuity loss，记为 L_c。

    含义：
    相邻时间的 intent 不应该剧烈跳变，
    因为论文把 intent 定义成“短时间窗口内相对稳定的未来行为趋势”。
    """
    cos_sim = F.cosine_similarity(intent_prev, intent_now, dim=-1, eps=eps)
    return (-cos_sim).mean()


def kl_intent_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """计算论文里的 KL 损失，记为 L_k。

    作用：
    continuity loss 会推动 intent 更稳定，
    但太强会让分布塌缩、探索不足。
    所以这里用 KL 把 intent 分布往标准正态附近拉，
    保持一定多样性。

    这里采用常见 VAE 写法：
        KL(N(mu, sigma^2) || N(0, I))
      = -1/2 * sum(1 + logvar - mu^2 - exp(logvar))
    """
    kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
    return kl.sum(dim=-1).mean()


def total_intent_loss(
    cfg: CoDeConfig,
    pred_actions: torch.Tensor,
    target_actions: torch.Tensor,
    intent_prev: torch.Tensor,
    intent_now: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """计算论文里的 L_int。

    L_int = lambda_inf * L_inf + lambda_c * L_c + lambda_k * L_k
    """
    L_inf = inference_loss(pred_actions, target_actions, cfg.discrete_action)
    L_c = continuity_loss(intent_prev, intent_now, cfg.eps)
    L_k = kl_intent_loss(mu, logvar)

    L_int = (
        cfg.lambda_inf * L_inf
        + cfg.lambda_c * L_c
        + cfg.lambda_k * L_k
    )

    return {
        "L_inf": L_inf,
        "L_c": L_c,
        "L_k": L_k,
        "L_int": L_int,
    }


def total_training_loss(
    rl_loss: torch.Tensor,
    intent_loss_dict: Dict[str, torch.Tensor],
    alignment_loss: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """组合成论文里的总损失。

    L_total = L_RL + L_int + L_e

    参数：
    - rl_loss:
      你自己的强化学习方法算出来的损失
    - intent_loss_dict:
      `total_intent_loss(...)` 的返回值
    - alignment_loss:
      dual alignment 里的 L_e
    """
    L_int = intent_loss_dict["L_int"]
    L_e = alignment_loss
    L_total = rl_loss + L_int + L_e

    return {
        "L_rl": rl_loss,
        "L_inf": intent_loss_dict["L_inf"],
        "L_c": intent_loss_dict["L_c"],
        "L_k": intent_loss_dict["L_k"],
        "L_int": L_int,
        "L_e": L_e,
        "L_total": L_total,
    }
