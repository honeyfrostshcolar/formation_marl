import torch
import torch.nn as nn


def build_mlp(in_dim, hidden_dims, out_dim, activation=nn.ReLU, activate_last=False):
    layers = []
    prev = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(activation())
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    if activate_last:
        layers.append(activation())
    return nn.Sequential(*layers)


def ensure_action_tensor(prev_action, batch_size, n_agents, action_dim, device, dtype):
    """统一处理 prev_action=None / numpy / tensor 三种情况。"""
    if prev_action is None:
        return torch.zeros(batch_size, n_agents, action_dim, device=device, dtype=dtype)
    if not torch.is_tensor(prev_action):
        prev_action = torch.as_tensor(prev_action, device=device, dtype=dtype)
    prev_action = prev_action.to(device=device, dtype=dtype)
    if prev_action.dim() == 2 and prev_action.shape == (n_agents, action_dim):
        prev_action = prev_action.unsqueeze(0)
    return prev_action
