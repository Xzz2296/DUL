import torch.nn as nn
import torch.nn.functional as F
import torch
class erf_loss(nn.Module):
    def __init__(self,softmax_head) -> None:
        super(erf_loss,self).__init__()
        self.weight = softmax_head.weight

    def forward(self, mu, std, label):
        '''
        mu: batch * embedding_dim
        std: batch * embedding_dim
        label: batch
        onehot: batch * class
        weight: class * embedding_dim
        '''
        # 利用广播机制
        mu_expanded = mu.unsqueeze(1).float() # batch * 1 * dim
        std_expanded = std.unsqueeze(1).float() # batch * 1 * dim
        weight = self.weight # C * dim
        weight_expanded = weight.unsqueeze(0)# 1 * C * dim
        dist =torch.abs(weight_expanded - mu_expanded) # batch * C * dim
        y = dist / torch.sqrt(torch.tensor(2.0))/(std_expanded + 1e-8) # batch * C * dim ; Y ~ N(0,1)

        one_hot_label = F.one_hot(label,num_classes=weight.shape[0]).unsqueeze(-1).float() # batch * C *1
        positive = one_hot_label * y
        negative = (1-one_hot_label) * y

        erf_value = torch.erf(positive)
        erfc_value = torch.erfc(negative)
        loss = torch.mean(torch.sum(erf_value,dim=-1)) + torch.mean(torch.sum(erfc_value,dim=-1))
        # loss =erf_value.mean() + erfc_value.mean()
        return loss

        