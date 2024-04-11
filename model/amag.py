import torch 
import torch.nn as nn 
from model.spatial_interaction import SI 
from model.gru import GRU
from model.transformer import * 


class AMAG(nn.Module):
    def __init__(self, channel_in, channel_out, hidden_size, num_channels=89, num_propagate=1, mode = 'train', corr=None, device='cpu') -> None:
        super().__init__()
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.hidden_size = hidden_size
        self.num_channels = num_channels
        self.hidden_enc = None 
        self.hidden_dec = None 
        self.num_propagate = num_propagate
        self.te = GRU(channel_in=self.channel_in, hidden_dim=self.hidden_size)  # Temporal Encoding Module
        self.si = SI(channel_in=hidden_size, channel_out=hidden_size, num_channels=self.num_channels, mode = mode, corr=corr)  # Spatial Interaction Module
        self.tr = GRU(channel_in=hidden_size, hidden_dim=channel_out)  # Temporal Readout Module 
        self.device = device
    def forward(self, x): 
        out, hidden = self.te(x, self.hidden_enc)
        self.hidden_enc = hidden.clone().detach()
        out = self.si(out)
        pred, hidden_dec = self.tr(out, self.hidden_dec)
        self.hidden_dec = hidden_dec.clone().detach()
        return pred 


class AMAG_transformer(nn.Module):
    def __init__(self, channel_in, channel_out, hidden_size, num_channels=89, num_propagate=1, mode = 'train', corr=None, device='cpu') -> None:
        super().__init__()
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.hidden_size = hidden_size
        self.num_channels = num_channels
        self.num_propagate = num_propagate
        mask = torch.ones(19) 
        mask = torch.concatenate([mask, torch.zeros(1)]).unsqueeze(-1).unsqueeze(0)
        self.register_buffer('mask', mask)
        
        self.te = TemporalLayer(channel_in, hidden_size, hidden_size, 20, 4, 0.1)
        self.si = SI(channel_in=hidden_size, channel_out=hidden_size, num_channels=self.num_channels, mode = mode, corr=corr)  # Spatial Interaction Module
        self.tr = TemporalLayer(hidden_size, hidden_size, hidden_size, 20, 4, 0.1)  # Temporal Readout Module 
        self.fc = nn.Linear(self.hidden_size, self.channel_out)
        self.device = device
    def forward(self, x): 
        out = torch.zeros(size=(x.shape[0], x.shape[1], x.shape[2], self.hidden_size)).to(x.device)
        for i in range(x.shape[2]):
            out[:,:,i,:] = self.te(x[:, :, i, :], self.mask)
        for j in range(x.shape[1]):
            out[:,j,:,:] = self.si(out[:, j, :, :])
        for k in range(x.shape[2]):
            out[:,:,k,:] = self.tr(out[:,:,k,:], self.mask)
        return self.fc(out)


if __name__ == '__main__':
    x = torch.randn(size=(32, 20, 89, 9))
    x = x.to(torch.device('cuda'))
    amag = AMAG_transformer(channel_in=x.shape[-1], 
                channel_out=9, 
                hidden_size=128, 
                num_channels=x.shape[-2],
                num_propagate=1).to(torch.device('cuda'))
    print(amag(x).shape)