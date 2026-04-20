from typing import Dict

import torch
import torch.nn.functional as F


def inference_loss(
    pred_actions: torch.Tensor,
    target_actions: torch.Tensor,
) -> torch.Tensor:
    """连续动作场景下的未来动作推断损失 L_inf。

    当前工程是连续动作控制：
    - pred_actions: [B, K, A]
    - target_actions: [B, K, A]

    所以这里直接使用 MSE，不再保留离散动作 cross-entropy 分支。
    """
    return F.mse_loss(pred_actions, target_actions)


def continuity_loss(
    intent_prev: torch.Tensor,
    intent_now: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """连续性损失 L_c。

    约束相邻时刻的 intent 不要剧烈跳变，
    保持“短时间内行为趋势相对稳定”。
    """
    cos_sim = F.cosine_similarity(intent_prev, intent_now, dim=-1, eps=eps)
    return (-cos_sim).mean()


def kl_intent_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL 损失 L_k。

    约束意图分布不要塌缩到单点，
    保持一定的分布平滑性和表达能力。
    """
    kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
    return kl.sum(dim=-1).mean()


def total_intent_loss(
    args,
    pred_actions: torch.Tensor,
    target_actions: torch.Tensor,
    intent_prev: torch.Tensor,
    intent_now: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """计算意图学习总损失 L_int。

    L_int = lambda_inf * L_inf + lambda_c * L_c + lambda_k * L_k
    """
    L_inf = inference_loss(pred_actions, target_actions)
    L_c = continuity_loss(intent_prev, intent_now, args.eps)
    L_k = kl_intent_loss(mu, logvar)

    L_int = (
        args.lambda_inf * L_inf
        + args.lambda_c * L_c
        + args.lambda_k * L_k
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
    belief_loss: torch.Tensor,
    alignment_loss: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """总损失。

    L_total = L_RL + L_int + L_e
    """
    L_int = intent_loss_dict["L_int"]
    L_e = alignment_loss
    L_b = belief_loss
    L_total = rl_loss + L_int + L_e + L_b

    return {
        "L_rl": rl_loss,
        "L_inf": intent_loss_dict["L_inf"],
        "L_c": intent_loss_dict["L_c"],
        "L_k": intent_loss_dict["L_k"],
        "L_int": L_int,
        "L_e": L_e,
        "L_b": L_b,
        "L_total": L_total,
    }