import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from models.magic_maddpg import MAGICCoDeActor
from models.magic_maddpg import Centralized_Critic
from utils.code_losses import total_intent_loss, total_training_loss


class MADDPG_Agent:
    """[MOD 9] 整理后的 Agent。

    核心变化:
    - select_action 使用 runtime delay buffer
    - update 同时优化 RL loss + CoDe auxiliary losses
    - 训练时 receiver-side 使用 replay 中保存的 comm snapshot
    """

    def __init__(self, args):
        self.args = args
        self.num_followers = args.num_followers
        self.gamma = args.gamma
        self.tau = args.tau
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = MAGICCoDeActor(args).to(self.device)
        self.target_actor = MAGICCoDeActor(args).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.lr_actor)

        self.critic = Centralized_Critic(args.num_followers, args.obs_size, args.action_dim).to(self.device)
        self.target_critic = Centralized_Critic(args.num_followers, args.obs_size, args.action_dim).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.lr_critic)

    def reset_runtime(self):
        self.actor.reset_runtime_state(batch_size=1)
        self.target_actor.reset_runtime_state(batch_size=1)

    def init_hidden(self):
        h = torch.zeros(1, self.num_followers, self.args.hid_size, device=self.device)
        c = torch.zeros(1, self.num_followers, self.args.hid_size, device=self.device)
        return h, c

    def select_action(self, obs_array, h_in, c_in, add_noise=True, noise_scale=0.15):
        obs_tensor = torch.as_tensor(obs_array, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.actor.eval()
        with torch.no_grad():
            action_tensor, hard_weights, soft_weights, (h_out, c_out), aux = self.actor(
                obs_tensor,
                (h_in, c_in),
                prev_action=None,
                runtime_mode=True,
                external_comm=None,
            )
        self.actor.train()

        action = action_tensor.squeeze(0).cpu().numpy()
        action_policy = action_tensor.squeeze(0).cpu().numpy()
        action_exec = action_policy.copy()
        graphs = hard_weights.squeeze(0).cpu().numpy()
        graphs_soft = soft_weights.squeeze(0).cpu().numpy()

        if add_noise:
            noise = np.random.normal(0, noise_scale, size=action_exec.shape)
            action_exec = np.clip(action_exec + noise, -1.0, 1.0)

        # [MOD 9-1] 把 rollout 时真正使用的 receiver-side snapshot 一并返回，方便存入 buffer
        comm_snapshot = {
            "prev_action": aux["prev_action_used"].squeeze(0).cpu().numpy(),
            "route_hard": aux["route_hard"].squeeze(0).cpu().numpy(),
            "route_soft": aux["route_soft"].squeeze(0).cpu().numpy(),
            "sender_intents_recv": aux["sender_intents_recv"].squeeze(0).cpu().numpy(),
            "sender_hidden_recv": aux["sender_hidden_recv"].squeeze(0).cpu().numpy(),
            "recv_mask": aux["recv_mask"].squeeze(0).cpu().numpy(),
            "time_lags": aux["time_lags"].squeeze(0).cpu().numpy(),
        }
        return action_policy, action_exec, graphs, graphs_soft, h_out, c_out, comm_snapshot

    def update(self, replay_buffer):
        batch = replay_buffer.sample_transitions(self.args.batch_size)
        bsz = batch["obs"].size(0)

        global_obs = batch["obs"].reshape(bsz, -1)
        global_next_obs = batch["next_obs"].reshape(bsz, -1)
        global_actions = batch["action"].reshape(bsz, -1)

        # --------------------
        # 1) Critic
        # --------------------
        with torch.no_grad():
            next_actions, _, _, _, _ = self.target_actor(
                batch["next_obs"],
                (batch["h_out"], batch["c_out"]),
                prev_action=batch["action"],
                runtime_mode=False,
                external_comm=None,  # [MOD 9-2] 目标策略这里采用同步近似，避免再引入 next-step buffer 复杂度
            )
            target_q = self.target_critic(global_next_obs, next_actions.reshape(bsz, -1))
            target_q_val = batch["reward"] + (1.0 - batch["done"]) * self.gamma * target_q

        current_q = self.critic(global_obs, global_actions)
        critic_loss = F.mse_loss(current_q, target_q_val)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # --------------------
        # 2) Actor RL + receiver-side comm snapshot
        # --------------------
        external_comm = {
            "sender_intents": batch["sender_intents_recv"],
            "sender_hidden": batch["sender_hidden_recv"],
            "recv_mask": batch["recv_mask"],
            "time_lags": batch["time_lags"],
        }
        curr_actions, _, _, _, aux = self.actor(
            batch["obs"],
            (batch["h_in"], batch["c_in"]),
            prev_action=batch["prev_action"],
            runtime_mode=False,
            external_comm=external_comm,
        )
        actor_rl_loss = -self.critic(global_obs, curr_actions.reshape(bsz, -1)).mean()

        # --------------------
        # 3) CoDe auxiliary losses 辅助损失
        # --------------------
        num_valid_anchors = replay_buffer.num_valid_intent_anchors(self.args.pred_horizon)

        if num_valid_anchors >= self.args.batch_size:
            seq = replay_buffer.sample_intent_sequences(
                self.args.batch_size,
                self.args.pred_horizon
            )

            mu_prev, logvar_prev, intent_prev = self.actor.intent_encoder(
                seq["h_prev"],
                seq["prev_action_prev"]
            )
            mu_curr, logvar_curr, intent_curr = self.actor.intent_encoder(
                seq["h_curr"],
                seq["prev_action_curr"]
            )

            bn = self.args.batch_size * self.num_followers

            pred_actions = self.actor.intent_decoder(
                intent_curr.reshape(bn, -1),
                seq["h_curr"].reshape(bn, -1),
                seq["obs_decode"].reshape(bn, self.args.pred_horizon, self.args.obs_size),
                seq["prev_action_curr"].reshape(bn, self.args.action_dim),
            )

            target_actions = seq["target_action_seq"].reshape(
                bn, self.args.pred_horizon, self.args.action_dim
            )

            intent_loss_dict = total_intent_loss(
                self.args,
                pred_actions=pred_actions,
                target_actions=target_actions,
                intent_prev=intent_prev.reshape(bn, -1),
                intent_now=intent_curr.reshape(bn, -1),
                mu=mu_curr.reshape(bn, -1),
                logvar=logvar_curr.reshape(bn, -1),
            )
        else:
            zero = actor_rl_loss.new_zeros(())
            intent_loss_dict = {
                "L_inf": zero,
                "L_c": zero,
                "L_k": zero,
                "L_int": zero,
            }

        total_losses = total_training_loss(actor_rl_loss, intent_loss_dict, aux["L_e"])

        self.actor_optimizer.zero_grad()
        total_losses["L_total"].backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # --------------------
        # 4) soft update
        # --------------------
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        metrics = {
            "actor_loss": float(total_losses["L_total"].item()),
            "critic_loss": float(critic_loss.item()),
            "rl_loss": float(total_losses["L_rl"].item()),
            "L_inf": float(total_losses["L_inf"].item()),
            "L_c": float(total_losses["L_c"].item()),
            "L_k": float(total_losses["L_k"].item()),
            "L_e": float(total_losses["L_e"].item()),
        }
        return metrics
