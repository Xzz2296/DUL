import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from config import Backbone_Dict, dul_args_func
from head.metrics import ArcFace, CosFace, SphereFace, Am_softmax, Softmax
from loss.focal import FocalLoss
from util.utils import make_weights_for_balanced_classes, separate_irse_bn_paras, \
                       warm_up_lr, schedule_lr, get_time, AverageMeter, accuracy, add_gaussian_noise
from torch.nn import Parameter

from tensorboardX import SummaryWriter, writer
import os
import time
import numpy as np
from PIL import Image
import random



class My_LOSS(nn.Module):
    def __init__(self, in_features, out_features):
        super(My_LOSS, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        # self.weight = torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]])
    
    def calculate_prob(self, x, mu, std):
        x_expanded = x.unsqueeze(1)
        mu_expanded = mu.unsqueeze(1)
        std_expanded = std.unsqueeze(1)

        output = (-0.5* torch.log(2* torch.pi * std_expanded)- (x_expanded -mu_expanded)**2 )/2* std_expanded**2/ torch.sum()
        return output
    

    def forward(self, x, mu, std, label):
        # out = F.linear(x, self.weight)
        # output = torch.zeros(mu.shape[0], self.weight.shape[0])
        # for i in range(mu.shape[0]):
        #     for j in range(self.weight.shape[0]):
        #         output[i, j] = -torch.sum(((mu[i] - self.weight[j]) ** 2)) / torch.sum((std[i] ** 2)) - torch.log(torch.sum(std[i] ** 2))
        x_expanded = x.unsqueeze(1)
        mu_expanded = mu.unsqueeze(1)
        std_expanded = std.unsqueeze(1)
        # output = (-0.5* torch.log(2* torch.pi * std_expanded)- (x_expanded -mu_expanded)**2 )/2* std_expanded**2/ torch.sum()
        output = -torch.sum((mu_expanded - self.weight) ** 2, dim=2) / torch.sum(std_expanded ** 2, dim=2) - torch.log(torch.sum(std_expanded ** 2, dim=2))

        one_hot = torch.zeros(output.size())
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (output * one_hot) + (one_hot - 1.0) * output
        return output

class erf_loss(nn.Module):
    def __init__(self,softmax_head) -> None:
        super(erf_loss,self).__init__()
        self.weight = softmax_head.weight

    def forward(self, x, mu, std, label):
        '''
        x: batch * embedding_dim
        mu: batch * embedding_dim
        std: batch * embedding_dim
        label: batch
        onehot: batch * class
        weight: class * embedding_dim
        '''
        weight = self.weight # C * dim
        one_hot_label = F.one_hot(label,num_classes=weight.shape[0]).float()
        wy = torch.mm(one_hot_label, weight)
        wi = torch.mm(1-one_hot_label, weight)
        erf_value = torch.erf(abs(torch.dist(wy,mu))/ (torch.sqrt(torch.tensor(2))*std))
        erfc_value = torch.erfc(abs(torch.dist(wi,mu))/ (torch.sqrt(torch.tensor(2))*std))
        loss =erf_value.sum() + erfc_value.sum()
        
        return one_hot_label



class My_DUL:
    def __init__(self, dul_args):
        self.dul_args = dul_args

    def _model_loader(self, num_class):

        HEAD = ArcFace(in_features = self.dul_args.embedding_size, out_features = num_class, device_id = None, s=self.dul_args.arcface_scale)

        Loss_Dict = {
            'Focal': FocalLoss(),
            'Softmax': nn.CrossEntropyLoss()
        }
        LOSS = Loss_Dict[self.dul_args.loss_name]
        return HEAD, LOSS

    def _dul_runner(self):
        # Load model
        # HEAD, LOSS = self._model_loader(10575)
        # inputs = torch.load('inputs.pt', map_location=torch.device('cpu'))
        # labels = torch.load('labels.pt', map_location=torch.device('cpu'))
        # mu_dul = torch.load('mu_dul.pt', map_location=torch.device('cpu'))
        # std_dul = torch.load('std_dul.pt', map_location=torch.device('cpu'))
        HEAD =Softmax(in_features=3, out_features=3,device_id=None)
        LOSS =erf_loss(HEAD)
        # LOSS = My_LOSS(3, 4)
        input = torch.FloatTensor([[1.0, 2, 3], [4, 5, 6]])
        labels = torch.tensor([1,2])
        mu_dul = torch.tensor([[1, 2, 3], [4, 5, 6]])
        std_dul = torch.tensor([[0.5, 0.5, 1], [1, 1, 2]])

        # epsilon = torch.randn_like(std_dul)
        # features = mu_dul + epsilon * std_dul
        # variance_dul = std_dul**2

        # LOSS = LOSS.cuda()
        # outputs = LOSS(input,mu_dul, std_dul, labels)
        outputs = HEAD(input,labels)
        loss_head =LOSS(input,mu_dul,std_dul,labels)
        print('loss_head:', outputs)

if __name__ == '__main__':
    dul_train = My_DUL(dul_args_func())
    dul_train._dul_runner()