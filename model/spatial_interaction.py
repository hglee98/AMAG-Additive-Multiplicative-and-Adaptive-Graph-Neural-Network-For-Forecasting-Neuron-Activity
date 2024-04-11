import torch_geometric.nn as pygnn 
import torch.nn as nn 
from torch_geometric.utils import dense_to_sparse
from model.networks import Adaptor, MLP
import torch 
import numpy as np 

class AddConv(pygnn.MessagePassing):
    def __init__(self, channel_in, num_channels=89, corr=None, mode='train'):
        super().__init__(aggr='sum')
        self.channel_in = channel_in
        hidden_dim = self.channel_in//2
        self.num_channels = num_channels
        self.adaptor = Adaptor(self.channel_in, [self.channel_in, hidden_dim, 1])
        self.adj = nn.Parameter(torch.randn(num_channels, num_channels), requires_grad=True) if corr is None and mode == 'train' else nn.Parameter(torch.FloatTensor(corr), requires_grad=True)
        

    def forward(self, x):
        edge_index, edge_attr = dense_to_sparse(self.adj)
        s = self.adaptor(x, edge_index)
        out = self.propagate(edge_index, h=x.reshape(self.num_channels, -1), s=s, a=edge_attr.unsqueeze(-1))
        return out.reshape(-1, self.num_channels, x.shape[-1])

    def message(self, h_j, a, s):
        return s * a * h_j

class ModulatorConv(pygnn.MessagePassing):
    def __init__(self, channel_in, num_channels=89, corr=None, mode='train'):
        super().__init__(aggr='sum')
        self.channel_in = channel_in
        self.num_channels = num_channels
        self.adj = nn.Parameter(torch.randn(num_channels, num_channels), requires_grad=True) if corr is None and mode == 'train' else nn.Parameter(torch.FloatTensor(corr), requires_grad=True)
        
    def forward(self, x):
        edge_index, edge_attr = dense_to_sparse(self.adj)
        out = self.propagate(edge_index, h=x.reshape(self.num_channels, -1), a=edge_attr.unsqueeze(-1))
        return out.reshape(-1, self.num_channels, x.shape[-1])

    def message(self, h_j, h_i, a):
        return a * h_j * h_i


class SI(nn.Module):
    def __init__(self, channel_in, channel_out, num_channels=89, mode='train', corr=None):
        super().__init__()
        self.channel_in = channel_in 
        self.channel_out = channel_out
        self.add_conv = AddConv(self.channel_in, num_channels, mode=mode, corr=corr)
        self.modulator_conv = ModulatorConv(self.channel_in, num_channels, mode=mode, corr=corr)
        self.mlp_add = MLP(self.channel_in, [self.channel_in, self.channel_out])
        self.mlp_mod = MLP(self.channel_in, [self.channel_in, self.channel_out])
    def forward(self, data): 
        a_t = self.add_conv(data)
        m_t = self.modulator_conv(data)
        return (data + self.mlp_add(a_t) + self.mlp_mod(m_t)) / 3

if __name__ == '__main__':
    input_tensor = torch.randn(size=(32, 89, 32))
    gnn = SI(channel_in=32, channel_out=32, num_channels=input_tensor.shape[1])
    print(gnn(input_tensor).shape)