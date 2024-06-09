import torch.nn as nn
import torch
class density_loss(nn.Module):
    def __init__(self):
        super(density_loss, self).__init__()
    
    def forward(self,input,label):
        target = label[:, None] == label[None, :]
        log_input = torch.log(input + 1e-8)
        return torch.mean(torch.sum(-target * log_input, dim=1))


        