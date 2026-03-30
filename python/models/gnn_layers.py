import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
    
class GraphAttention(nn.Module):
    """
    Graph-Attentional layer used in MAGIC that can process differentiable communication graphs
    """

    def __init__(self, in_features, out_features, dropout, negative_slope, num_heads=1, bias=True, self_loop_type=2, average=False, normalize=False):
        super(GraphAttention, self).__init__()
        """
        Initialization method for the graph-attentional layer

        Arguments:
            in_features (int): number of features in each input node
            out_features (int): number of features in each output node
            dropout (int/float): dropout probability for the coefficients
            negative_slope (int/float): control the angle of the negative slope in leakyrelu
            number_heads (int): number of heads of attention
            bias (bool): if adding bias to the output
            self_loop_type (int): 0 -- force no self-loop; 1 -- force self-loop; other values (2)-- keep the input adjacency matrix unchanged
            average (bool): if averaging all attention heads
            normalize (bool): if normalizing the coefficients after zeroing out weights using the communication graph

            in_features (int): 每个输入节点中的特征数量
            out_features (int): 每个输出节点中的特征数量
            dropout (int/float): 系数的随机失活概率
            negative_slope (int/float): 控制 LeakyReLU 中负斜率的角度
            number_heads (int): 注意力头的数量
            bias (bool): 是否在输出中添加偏置
            self_loop_type (int): 0 —— 强制无自环；1 —— 强制添加自环；其他值（2）—— 保持输入邻接矩阵不变
            average (bool): 是否对所有注意力头的结果进行平均
            normalize (bool): 在使用通信图将权重置零后，是否对系数进行归一化
        """

        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.num_heads = num_heads
        self.self_loop_type = self_loop_type
        self.average = average
        self.normalize = normalize

        self.W = nn.Parameter(torch.zeros(size=(in_features, num_heads * out_features)))
        self.a_i = nn.Parameter(torch.zeros(size=(num_heads, out_features, 1)))
        self.a_j = nn.Parameter(torch.zeros(size=(num_heads, out_features, 1)))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features if average else num_heads * out_features))
        else:
            self.register_parameter('bias', None)
        self.leakyrelu = nn.LeakyReLU(self.negative_slope)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """
        Initialization for the parameters of the graph-attentional layer
        """
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.W.data, gain=gain)
        nn.init.xavier_normal_(self.a_i.data, gain=gain)
        nn.init.xavier_normal_(self.a_j.data, gain=gain)
        if self.bias is not None:
            nn.init.zeros_(self.bias.data)

    def forward(self, x, adj):
        """
        [✅ 修复] 支持纯 Batched 运算!
        x: [B, N, in_features]
        adj: [B, N, N]
        """
        B, N, _ = x.shape

        # [B, N, in] @ [in, H*out] -> [B, N, H, out]
        h = torch.matmul(x, self.W).view(B, N, self.num_heads, self.out_features)

        # magic是有多头注意力机制的，每个头都有自己的权重和偏置，最后将所有头的输出进行拼接

        # 动态设备分配，避免 CPU Tensor 报错
        eye = torch.eye(N, N, device=x.device, dtype=x.dtype).unsqueeze(0).expand(B, N, N)
        ones = torch.ones(N, N, device=x.device, dtype=x.dtype).unsqueeze(0).expand(B, N, N)
        
        if self.self_loop_type == 0:
            adj = adj * (ones - eye)
        elif self.self_loop_type == 1:
            adj = eye + adj * (ones - eye)

        # 计算注意力打分 (einsum 高效实现)
        a_i = self.a_i.squeeze(-1) # [H, out]
        a_j = self.a_j.squeeze(-1)
        
        e_i = torch.einsum('bnhf,hf->bnh', h, a_i) # [B, N, H]
        e_j = torch.einsum('bnhf,hf->bnh', h, a_j)
        
        # 广播相加并激活
        scores = self.leakyrelu(e_i.unsqueeze(2) + e_j.unsqueeze(1)) # [B, N, N, H]

        # 彻底解决 Softmax 污染问题：先 mask 填入极小值，再 softmax
        mask = adj.unsqueeze(-1) > 0
        scores = scores.masked_fill(~mask, -1e9)

        attn = torch.softmax(scores, dim=2)
        attn = attn * mask.float()

        denom = attn.sum(dim=2, keepdim=True).clamp_min(1e-6)
        attn = attn / denom
        attn = F.dropout(attn, self.dropout, training=self.training)

        # 聚合特征: [B, N, N, H] * [B, N, H, out] -> [B, N, H, out]
        out = torch.einsum('bijh,bjhf->bihf', attn, h)

        if self.average:
            out = out.mean(dim=2) # [B, N, out]
        else:
            out = out.reshape(B, N, -1) # [B, N, H*out]
            
        if self.bias is not None:
            out = out + self.bias

        return out

    def __repr__(self):
        return self.__class__.__name__ + '(in_features={}, out_features={})'.format(self.in_features, self.out_features)
    
