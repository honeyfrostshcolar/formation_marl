import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from models.gnn_layers import GraphAttention

class MAGIC_Actor(nn.Module):
    """
    The communication protocol of Multi-Agent Graph AttentIon Communication (MAGIC)
    """
    def __init__(self, args):
        super(MAGIC_Actor, self).__init__()
        """
        Initialization method for the MAGIC communication protocol (2 rounds of communication)

        Arguements:
            args (Namespace): Parse arguments
        """

        self.args = args
        self.nagents = args.num_followers
        self.hid_size = args.hid_size
        self.n = args.num_followers
        
        dropout = 0
        negative_slope = 0.2

        # initialize sub-processors
        self.sub_processor1 = GraphAttention(args.hid_size, args.gat_hid_size, dropout=dropout, negative_slope=negative_slope, num_heads=args.gat_num_heads, self_loop_type=args.self_loop_type1, average=False, normalize=args.first_gat_normalize)
        self.sub_processor2 = GraphAttention(args.gat_hid_size*args.gat_num_heads, args.hid_size, dropout=dropout, negative_slope=negative_slope, num_heads=args.gat_num_heads_out, self_loop_type=args.self_loop_type2, average=True, normalize=args.second_gat_normalize)
        # initialize the gat encoder for the Scheduler
        if args.use_gat_encoder:
            self.gat_encoder = GraphAttention(args.hid_size, args.gat_encoder_out_size, dropout=dropout, negative_slope=negative_slope, num_heads=args.ge_num_heads, self_loop_type=1, average=True, normalize=args.gat_encoder_normalize)

        self.obs_encoder = nn.Linear(args.obs_size, args.hid_size)

        # self.init_hidden(args.batch_size)
        self.lstm_cell= nn.LSTMCell(args.hid_size, args.hid_size)

        # initialize mlp layers for the sub-schedulers
        if not args.first_graph_complete:
            if args.use_gat_encoder:
                self.sub_scheduler_mlp1 = nn.Sequential(
                    nn.Linear(args.gat_encoder_out_size*2, args.gat_encoder_out_size//2),
                    nn.ReLU(),
                    nn.Linear(args.gat_encoder_out_size//2, args.gat_encoder_out_size//2),
                    nn.ReLU(),
                    nn.Linear(args.gat_encoder_out_size//2, 2))
            else:
                self.sub_scheduler_mlp1 = nn.Sequential(
                    nn.Linear(self.hid_size*2, self.hid_size//2),
                    nn.ReLU(),
                    nn.Linear(self.hid_size//2, self.hid_size//8),
                    nn.ReLU(),
                    nn.Linear(self.hid_size//8, 2))
                
        if args.learn_second_graph and not args.second_graph_complete:
            if args.use_gat_encoder:
                self.sub_scheduler_mlp2 = nn.Sequential(
                    nn.Linear(args.gat_encoder_out_size*2, args.gat_encoder_out_size//2),
                    nn.ReLU(),
                    nn.Linear(args.gat_encoder_out_size//2, args.gat_encoder_out_size//2),
                    nn.ReLU(),
                    nn.Linear(args.gat_encoder_out_size//2, 2))
            else:
                self.sub_scheduler_mlp2 = nn.Sequential(
                    nn.Linear(self.hid_size*2, self.hid_size//2),
                    nn.ReLU(),
                    nn.Linear(self.hid_size//2, self.hid_size//8),
                    nn.ReLU(),
                    nn.Linear(self.hid_size//8, 2))

        if args.message_encoder:
            self.message_encoder = nn.Linear(args.hid_size, args.hid_size)
        if args.message_decoder:
            self.message_decoder = nn.Linear(args.hid_size, args.hid_size)

        # initialize weights as 0
        if args.comm_init == 'zeros':
            if args.message_encoder:
                self.message_encoder.weight.data.zero_()
            if args.message_decoder:
                self.message_decoder.weight.data.zero_()
            if not args.first_graph_complete:
                self.sub_scheduler_mlp1.apply(self.init_linear)
            if args.learn_second_graph and not args.second_graph_complete:
                self.sub_scheduler_mlp2.apply(self.init_linear)
                   
        # initialize the action head (in practice, one action head is used)
        self.action_head = nn.Sequential(
            nn.Linear(2 * self.hid_size, self.hid_size),
            nn.ReLU(),
            nn.Linear(self.hid_size, args.action_dim),
            nn.Tanh()  # 必须用 Tanh，适配 MADDPG 的连续动作
        )


    def forward(self, x):
        """ 
        MAGIC 的前向传播函数（两轮通信）

        参数：
        x (list): 通信协议的输入列表，包含 [观测值, (上一时刻隐藏状态, 上一时刻细胞状态)]
        observations (tensor): 所有智能体的观测值，维度为 [1（批量大小） * n * obs_size]
        previous hidden/cell states (tensor): 上一时刻的隐藏状态/细胞状态，维度为 [n * hid_size]

        返回值：
        action_out (list): 一个张量列表，每个张量维度为 [1（批量大小） * n * num_actions]，表示输出的策略分布
        value_head (tensor): 估计值，维度为 [n * 1]
        next hidden/cell states (tensor): 下一时刻的隐藏状态/细胞状态，维度为 [n * hid_size]
         
        """
        obs, extras = x
        h_in, c_in = extras # [B, N, H]
        
        B, N, _ = obs.shape
        encoded_obs = self.obs_encoder(obs) 
        
        # 兼容死掉的 agent (全 1 代表全活)
        agent_mask = torch.ones(B, N, 1, device=obs.device, dtype=obs.dtype)

        # LSTM 时序更新 (展平 -> 计算 -> 还原)
        h_out, c_out = self.lstm_cell(
            encoded_obs.view(B * N, -1), 
            (h_in.view(B * N, -1), c_in.view(B * N, -1))
        )
        h_out = h_out.view(B, N, self.hid_size)
        c_out = c_out.view(B, N, self.hid_size)

        comm = h_out
        if self.args.message_encoder:
            comm = self.message_encoder(comm)
        comm = comm * agent_mask
        comm_ori = comm.clone()

        # ======== 第一轮通信 ========
        if not self.args.first_graph_complete:
            if self.args.use_gat_encoder:
                adj_comp = self.get_complete_graph(agent_mask)
                enc_st1 = self.gat_encoder(comm, adj_comp)
                adj1_hard, _ = self.sub_scheduler(self.sub_scheduler_mlp1, enc_st1, agent_mask, self.args.directed)
            else:
                adj1_hard, _ = self.sub_scheduler(self.sub_scheduler_mlp1, comm, agent_mask, self.args.directed)
        else:
            adj1_hard = self.get_complete_graph(agent_mask)

        # 三维张量直接过 GAT
        comm = F.elu(self.sub_processor1(comm, adj1_hard))
        
        # ======== 第二轮通信 ========
        if self.args.learn_second_graph and not self.args.second_graph_complete:
            if self.args.use_gat_encoder:
                enc_st2 = self.gat_encoder(comm_ori, self.get_complete_graph(agent_mask)) if self.args.first_graph_complete else enc_st1
                adj2_hard, adj2_soft = self.sub_scheduler(self.sub_scheduler_mlp2, enc_st2, agent_mask, self.args.directed)
            else:
                adj2_hard, adj2_soft = self.sub_scheduler(self.sub_scheduler_mlp2, comm_ori, agent_mask, self.args.directed)
        elif not self.args.learn_second_graph and not self.args.second_graph_complete:
            adj2_hard, adj2_soft = adj1_hard, torch.zeros_like(adj1_hard)
        else:
            adj2_hard = self.get_complete_graph(agent_mask)
            adj2_soft = adj2_hard.clone()
            
        comm = self.sub_processor2(comm, adj2_hard)
        comm = comm * agent_mask
        
        if self.args.message_decoder:
            comm = self.message_decoder(comm)

        # 输出动作
        combined_feat = torch.cat((h_out, comm), dim=-1)
        actions_out = self.action_head(combined_feat)

        # 返回去掉了 value_head
        return actions_out, adj2_hard, adj2_soft, (h_out, c_out)

    def get_agent_mask(self, batch_size, n, info):
        """
        Function to generate agent mask to mask out inactive agents (only effective in Traffic Junction)

        Returns:
            num_agents_alive (int): number of active agents
            agent_mask (tensor): [n, 1]
        """

        if 'alive_mask' in info:
            agent_mask = torch.from_numpy(info['alive_mask'])
            num_agents_alive = agent_mask.sum()
        else:
            agent_mask = torch.ones(n)
            num_agents_alive = n

        agent_mask = agent_mask.view(n, 1).clone()

        return num_agents_alive, agent_mask

    def init_linear(self, m):
        """
        Function to initialize the parameters in nn.Linear as o 
        """
        if type(m) == nn.Linear:
            m.weight.data.fill_(0.)
            m.bias.data.fill_(0.)
        
    def init_hidden(self, batch_size, n=None):


        n_agents = n if n is not None else self.nagents
        return tuple(( torch.zeros(batch_size * n_agents, self.hid_size, requires_grad=True),
                       torch.zeros(batch_size * n_agents, self.hid_size, requires_grad=True)))
    
    
    def sub_scheduler(self, mlp, hidden_state, agent_mask, directed=True):
        """ 完全 Batched 化的高效调度器 """
        B, N, F_dim = hidden_state.shape

        hi = hidden_state.unsqueeze(2).expand(B, N, N, F_dim)
        hj = hidden_state.unsqueeze(1).expand(B, N, N, F_dim)
        pair_feat = torch.cat([hi, hj], dim=-1) # [B, N, N, 2F]

        logits = mlp(pair_feat) # [B, N, N, 2]

        if not directed:
            logits = 0.5 * (logits + logits.transpose(1, 2))

        # 同时返回硬图和软图
        hard = F.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1)[..., 1]
        soft = torch.softmax(logits, dim=-1)[..., 1]

        mask = torch.bmm(agent_mask, agent_mask.transpose(1, 2)) # [B, N, N]
        hard = hard * mask
        soft = soft * mask

        return hard, soft
    
    def get_complete_graph(self, agent_mask):
        # agent_mask: [B, N, 1]
        return torch.bmm(agent_mask, agent_mask.transpose(1, 2))
    

class Centralized_Critic(nn.Module):
    """
    上帝视角评论家 (集中式 Critic)
    输入：所有人的观测 + 所有人的动作
    输出：一个全局 Q 值打分
    """
    def __init__(self, num_followers, obs_dim, action_dim):
        super(Centralized_Critic, self).__init__()
        # Critic 需要看全图，所以输入维度是 N 个人的 obs 和 N 个人的 action 拼接在一起
        self.global_obs_dim = num_followers * obs_dim
        self.global_action_dim = num_followers * action_dim
        
        self.fc1 = nn.Linear(self.global_obs_dim + self.global_action_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1) # 输出一个 Q 值

    def forward(self, global_obs, global_actions):
        # 将全局状态和全局动作拼接
        x = torch.cat([global_obs, global_actions], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value