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


class Softmax(nn.Module):
    r"""Implement of Softmax (normal classification head):
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            device_id: the ID of GPU where the model will be trained by model parallel.
                       if device_id=None, it will be trained on CPU without model parallel.
        """
    def __init__(self, in_features, out_features, device_id):
        super(Softmax, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device_id = device_id

        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        self.bias = Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, label):
        if self.device_id == None:
            # out = F.linear(x, self.weight, self.bias)
            out = F.linear(x, self.weight)
        else:
            sub_weights = torch.chunk(self.weight, len(self.device_id), dim=0)
            sub_biases = torch.chunk(self.bias, len(self.device_id), dim=0)
            temp_x = x.cuda(self.device_id[0])
            weight = sub_weights[0].cuda(self.device_id[0])
            bias = sub_biases[0].cuda(self.device_id[0])
            # out = F.linear(temp_x, weight, bias)
            out = F.linear(x, self.weight)
            for i in range(1, len(self.device_id)):
                temp_x = x.cuda(self.device_id[i])
                weight = sub_weights[i].cuda(self.device_id[i])
                bias = sub_biases[i].cuda(self.device_id[i])
                out = torch.cat((out, F.linear(temp_x, weight, bias).cuda(self.device_id[0])), dim=1)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zeros_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zeros_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zeros_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zeros_()


class My_HEAD(nn.Module):
    def __init__(self):
        super(My_HEAD, self).__init__()

    def forward(self, x, mu, std):
        # output = torch.zeros(x.shape[0], mu.shape[0])
        # for i in range(x.shape[0]):
        #     for j in range(mu.shape[0]):
        #         output[i, j] = -torch.sum(((x[i] - mu[j]) ** 2)) / (2.0 * torch.sum((std[j] ** 2) + 10e-8)) - torch.log(torch.sum(std[j])) - 0.5 * torch.log(torch.tensor(2 * torch.pi))
        # 下面是对上面的代码进行优化，使用广播机制，output[i,j]的含义是第i个样本属于第j个高斯分布的概率密度
        x_expanded = x.unsqueeze(1)
        mu_expanded = mu.unsqueeze(0)
        std_expanded = std.unsqueeze(0)
        output = -torch.sum((x_expanded - mu_expanded) ** 2, dim=2) / (2.0 * torch.sum(std_expanded ** 2, dim=2) + 1e-8) - torch.log(torch.sum(std_expanded, dim=2)) -  0.5 * torch.log(torch.tensor(2 * torch.pi))
        output_softmax = torch.nn.functional.softmax(output, dim=1)
        return output_softmax


class My_Density(nn.Module):
    def __init__(self,Softmax_head):
        '''
        weight : class * embedding
        mu: batch * embedding
        std: batch * embedding
        '''
        super(My_HEAD, self).__init__()
        self.weight = Softmax_head.weight
    def forward(self, mu, std):
        w = self.weight
        weight_expanded = mu.unsqueeze(0) # 1 * class * dim
        mu_expanded = mu.unsqueeze(1) # batch * 1 * dim
        std_expanded = std.unsqueeze(1) # batch * 1 * dim


        output = -(weight_expanded - mu_expanded) ** 2 / (2.0 * torch.sum(std_expanded ** 2, dim=2) + 1e-8) - torch.log(torch.sum(std_expanded, dim=2)) -  0.5 * torch.log(torch.tensor(2 * torch.pi))
        output_softmax = torch.nn.functional.softmax(output, dim=1)
        return output_softmax


class My_LOSS(nn.Module):
    def __init__(self):
        super(My_LOSS, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input, label):
        ce_result =self.criterion(input, label)
        # 将label转换为one-hot编码
        target = label[:, None] == label[None, :]
        target = target.int()
        # print(target)
        # 求log_softmax
        # log_softmax = torch.nn.functional.log_softmax(input, dim=1)
        log_input = torch.log(input + 1e-8)
        return torch.mean(torch.sum(-target * log_input, dim=1))


class My_DUL:
    def __init__(self, dul_args):
        self.dul_args = dul_args

    def _dul_runner(self):
        # Load model
        # HEAD, LOSS = self._model_loader(10575)
        # inputs = torch.load('inputs.pt', map_location=torch.device('cpu'))
        # labels = torch.load('labels.pt', map_location=torch.device('cpu'))
        # mu_dul = torch.load('mu_dul.pt', map_location=torch.device('cpu'))
        # std_dul = torch.load('std_dul.pt', map_location=torch.device('cpu'))

        HEAD = My_HEAD()
        my_Softmax = Softmax(in_features=3, out_features=3,device_id=None)
        LOSS = My_LOSS()
        input = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        labels = torch.tensor([1,1,2])
        mu_dul = torch.tensor([[3, 4, 5], [6, 7, 8], [9, 10, 11]])
        std_dul = torch.tensor([[0.5, 0.5, 1], [1, 1, 2], [2, 2, 3]])

        # if torch.cuda.device_count() > 1:
        #     print("Let's use", torch.cuda.device_count(), "GPUs!")
        #     LOSS = nn.DataParallel(LOSS)

        # LOSS = LOSS.cuda()
        outputs = HEAD(input, mu_dul, std_dul)
        loss = LOSS(outputs, labels)
        print(loss)

if __name__ == '__main__':
    dul_train = My_DUL(dul_args_func())
    dul_train._dul_runner()