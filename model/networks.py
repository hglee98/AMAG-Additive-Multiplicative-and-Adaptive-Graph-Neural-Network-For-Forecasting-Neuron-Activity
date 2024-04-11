import torch 
import torch.nn as nn 
import torch_geometric as pyg 
from torch_geometric.utils.sparse import dense_to_sparse
from model.message_passing import MessagePassing2
from typing import List, Optional, Callable
import numpy as np 


class MLP(nn.Sequential): 
    def __init__(self, channel_in: int,
                 hidden_dims: List[int], 
                 norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
                 bias: bool = True,
                 dropout: float = 0.0,
                 activation: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
                 inplace: Optional[bool] = None,):
        
        layers = list()
        params = {} if inplace is None else {"inplace": inplace}
        for h in hidden_dims[:-1]:
            layers.append(torch.nn.Linear(channel_in, h, bias=bias))
            if norm_layer is not None: 
                layers.append(norm_layer(h))
            layers.append(activation(**params))
            layers.append(torch.nn.Dropout(p=dropout, **params)) 
            channel_in = h 
        
        layers.append(torch.nn.Linear(channel_in, hidden_dims[-1], bias=bias))
        layers.append(torch.nn.Dropout(p=dropout, **params))

        super().__init__(*layers)


class Adaptor(MessagePassing2): 
    def __init__(self, channel_in, hidden_dims): 
        super().__init__()
        self.channel_in = channel_in 
        self.hidden_dims = hidden_dims
        self.mlp = MLP(channel_in=channel_in*2, hidden_dims=hidden_dims)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, h, edge_index): 
        # h: [B, C, D] 
        _, s = self.propagate(h=h, edge_index=edge_index)
        return s[0,:,:] 
    def message(self, h_i, h_j):
        return self.sigmoid(self.mlp(torch.cat([h_i, h_j], dim=-1)))
    


if __name__ == '__main__':
    input_tensor = torch.randn(size=(32, 89, 32))
    adaptor = Adaptor(32, [32, 16, 1])
    adj = np.ones(shape=(89, 89))
    edge_index, edge_attr = dense_to_sparse(torch.Tensor(adj))
    print(edge_index.shape)
    print(adaptor(input_tensor, edge_index).shape)
