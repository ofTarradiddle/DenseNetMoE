import torch
import torch.nn as nn

class DenseBlock(nn.Module):
    def __init__(self, in_features, growth_rate, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(in_features + i * growth_rate, growth_rate) for i in range(num_layers)])
        
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x_i = layer(torch.cat([x, x_out], dim=1))
            x_out = x_out + x_i
        return x_out

class AFTSimple(nn.Module):
    def __init__(self, max_seqlen, dim, hidden_dim=64):
        super().__init__()
        '''
        max_seqlen: the maximum number of timesteps (sequence length) to be fed in
        dim: the embedding dimension of the tokens
        hidden_dim: the hidden dimension used inside AFT Full
        
        Number of Heads is 1 as done in the paper.
        '''
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.to_q = nn.Linear(dim, hidden_dim)
        self.to_k = nn.Linear(dim, hidden_dim)
        self.to_v = nn.Linear(dim, hidden_dim)
        self.project = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        B, T, _ = x.shape
        Q = self.to_q(x).view(B, T, self.hidden_dim)
        K = self.to_k(x).view(B, T, self.hidden_dim)
        V = self.to_v(x).view(B, T, self.hidden_dim)

        '''
        From the paper
        '''
        weights = torch.mul(torch.softmax(K, 1), V).sum(dim=1, keepdim=True)
        Q_sig = torch.sigmoid(Q)
        Yt = torch.mul(Q_sig, weights)

        Yt = Yt.view(B, T, self.hidden_dim)
        Yt = self.project(Yt)

        return Yt
        
 class DenseNetMoE(nn.Module):
    def __init__(self, dim, growth_rate, num_dense_blocks, num_layers_per_block, num_experts):
        super().__init__()
        self.num_experts = num_experts
        self.dense_blocks = nn.ModuleList([DenseBlock(dim, growth_rate, num_layers_per_block) for _ in range(num_dense_blocks)])
        self.gating = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList([AFTSimple(max_seqlen, dim) for _ in range(num_experts)])

    def forward(self, x):
        for dense_block in self.dense_blocks:
            x = dense_block(x)

        gating_weights = torch.softmax(self.gating(x), dim=1)
        expert_outputs = [expert(x) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=1)
        expert_outputs = torch.mul(gating_weights.unsqueeze(-1), expert_outputs)
        expert_outputs = expert_outputs.sum(dim=1)

        return expert_outputs
