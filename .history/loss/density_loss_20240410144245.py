import torch.nn as nn

class density_loss(nn.Module):
    def __init__(self,x,mu,std):
        super(density_loss, self).__init__()
        