import torch.nn as nn
import torch
class Density_loss(nn.Module):
    def __init__(self):
        super(Density_loss, self).__init__()
    
    def forward(self,input,label):
        target = label[:, None] == label[None, :]
        target = target.int()
        log_input = torch.log(input + 1e-8)
        return torch.mean(torch.sum(-target * log_input, dim=1))


        