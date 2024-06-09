import torch.nn as nn
import torch
class density_loss(nn.Module):
    def __init__(self):
        super(density_loss, self).__init__()
    
    def forward(self,input,label):

        log_input = torch.log(input + 1e-8)


        