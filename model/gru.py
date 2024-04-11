import torch 
import torch.nn as nn 


class GRU(torch.nn.Module):
    def __init__(self, channel_in, hidden_dim) -> None:
        super().__init__()
        self.channel_in = channel_in
        self.hidden_dim = hidden_dim 
        self.gru = nn.GRU(input_size=channel_in, hidden_size=hidden_dim, batch_first=True)
        
    def forward(self, x_t, hidden=None): 
        hidden = self.gru(x_t, hidden)
        return hidden



if __name__ == '__main__':
    x = torch.randn(size=(32, 20, 89, 9))
    encoder = GRU(9, 32)
    decoder = GRU(32, 9)
    hidden_enc = torch.zeros(size=(1,1,32), requires_grad=True)
    hidden_dec = torch.zeros(size=(1,1,32), requires_grad=True)
    for i in range(x.shape[1] - 1):
        out, hidden_enc = encoder(x[:, i, :, :], hidden_enc)
        out, hidden_dec = decoder(out, hidden_dec)