import torch.nn as nn

class density_loss(nn.Module):
    def __init__(self):
        super(density_loss, self).__init__()
    
    def forward(self,x,mu,std):

        x_expanded = x.unsqueeze(1)
        mu_expanded = mu.unsqueeze(0)
        std_expanded = std.unsqueeze(0)
        

        