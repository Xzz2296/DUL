from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math


# Support: ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']

class CircleLoss(nn.Module):
    def __init__(self, in_features, out_features, device_id, s = 256.0, m = 0.35):
        super(CircleLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device_id = device_id

        self.s = s
        self.O_p = 1 + m
        self.O_n = - m
        self.delta_p = 1 - m
        self.delta_n = m

        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        if self.device_id == None:
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        else:
            x = input
            sub_weights = torch.chunk(self.weight, len(self.device_id), dim=0)
            temp_x = x.cuda(self.device_id[0])
            weight = sub_weights[0].cuda(self.device_id[0])
            cosine = F.linear(F.normalize(temp_x), F.normalize(weight))
            for i in range(1, len(self.device_id)):
                temp_x = x.cuda(self.device_id[i])
                weight = sub_weights[i].cuda(self.device_id[i])
                cosine = torch.cat((cosine, F.linear(F.normalize(temp_x), F.normalize(weight)).cuda(self.device_id[0])), dim=1)

        scores = cosine
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size())

        alpha_p = (self.O_p - scores.detach()).clamp(min=0.)
        alpha_n = (scores.detach() - self.O_n).clamp(min=0.)

        one_hot = torch.zeros(scores.size())
        if self.device_id != None:
            one_hot = one_hot.cuda(self.device_id[0])
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        output = (one_hot * (alpha_p * (scores - self.delta_p)) + (1.0 - one_hot) * (alpha_n * (scores - self.delta_n)))
        output *= self.s
        return output


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
            out =F.linear(x,self.weight)
        else:
            sub_weights = torch.chunk(self.weight, len(self.device_id), dim=0)
            sub_biases = torch.chunk(self.bias, len(self.device_id), dim=0)
            temp_x = x.cuda(self.device_id[0])
            weight = sub_weights[0].cuda(self.device_id[0])
            bias = sub_biases[0].cuda(self.device_id[0])
            # out = F.linear(temp_x, weight, bias)
            out = F.linear(temp_x,weight)
            for i in range(1, len(self.device_id)):
                temp_x = x.cuda(self.device_id[i])
                weight = sub_weights[i].cuda(self.device_id[i])
                bias = sub_biases[i].cuda(self.device_id[i])
                # out = torch.cat((out, F.linear(temp_x, weight, bias).cuda(self.device_id[0])), dim=1)
                out = torch.cat((out, F.linear(temp_x, weight).cuda(self.device_id[0])), dim=1)
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


# class Density_Softmax(nn.Module):
#     '''
#     返回正态分布概率密度并取对数,方便后续计算
#     '''
#     def __init__(self):
#         super(Density_Softmax, self).__init__()

#     def forward(self, x, mu, std):
#         # output = torch.zeros(x.shape[0], mu.shape[0])
#         # for i in range(x.shape[0]):
#         #     for j in range(mu.shape[0]):
#         #         output[i, j] = -torch.sum(((x[i] - mu[j]) ** 2)) / (2.0 * torch.sum((std[j] ** 2) + 10e-8)) - torch.log(torch.sum(std[j])) - 0.5 * torch.log(torch.tensor(2 * torch.pi))
#         # 下面是对上面的代码进行优化，使用广播机制，output[i,j]的含义是第i个样本属于第j个高斯分布的概率密度
#         x_expanded = x.unsqueeze(1)
#         mu_expanded = mu.unsqueeze(0)
#         std_expanded = std.unsqueeze(0)
#         # density = -torch.sum((x_expanded - mu_expanded) ** 2, dim=2) / (2.0 * torch.sum(std_expanded ** 2, dim=2) + 1e-8) - torch.log(torch.sum(std_expanded, dim=2)) -  0.5 * torch.log(torch.tensor(2 * torch.pi))
#         density = -torch.sum((x_expanded - mu_expanded) ** 2 / (2.0 * std_expanded ** 2 + 1e-8), dim=2) - 0.5 * torch.log(torch.sum(std_expanded ** 2, dim=2) * 2.0 * torch.pi)
#         # output =torch.nn.functional.softmax(-torch.sum((x_expanded - mu_expanded) ** 2 / (2.0 * std_expanded ** 2 + 1e-8), dim=2) - 0.5 * torch.log(torch.sum(std_expanded ** 2, dim=2) * 2.0 * torch.pi),dim=1)
        
#         # output = torch.nn.functional.softmax(density, dim=1)
#         return density


class My_HEAD(nn.Module):
    def __init__(self, weight):
        super(My_HEAD, self).__init__()
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        # self.weight = weight
        self.bias = torch.log(torch.tensor(2.0 * torch.pi))

    def forward(self, mu, std):
        weight = self.weight
        mu = mu.float()  
        std = std.float()
        # 正确的运算
        # output = torch.zeros(mu.shape[0], weight.shape[0])
        # for i in range(mu.shape[0]):
        #     for j in range(weight.shape[0]):
        #         # output[i, j] = -0.5*(torch.sum(((mu[i] - weight[j]) ** 2) / (std[i] ** 2)) +  torch.log(torch.sum(std[i])**2))
        #         output[i, j] = torch.sum(((mu[i] - weight[j]) ** 2) / std[i]**2) +  torch.log(torch.sum(std[i])**2)

        # 优化后的运算
        # 计算mu^2/std^2 B*Dim B*dim -> B*1
        temp_mu = ((mu/std)**2).sum(dim=1).unsqueeze(1)
        # 计算weight^2/std^2 B*Dim C*dim -> B*C
        temp_weight = F.linear(torch.reciprocal(std)**2 , weight**2)
        # 计算mu*weight/std^2 B*dim C*dim -> B*C
        temp_mu_weight = 2 * F.linear(mu/std**2, weight)
        # density = temp_mu + temp_weight - temp_mu_weight + torch.log(torch.sum(std**2, dim=1)).unsqueeze(1)

        density = -0.5*(temp_mu + temp_weight - temp_mu_weight + torch.log(torch.sum(std**2, dim=1)).unsqueeze(1) +self.bias)

        # 不取log怎么算
        density = torch.exp(density)
        # 或
        density = torch.exp(-0.5*(temp_mu + temp_weight - temp_mu_weight)) / (self.bias * torch.sqrt(torch.sum(std**2, dim=1)).unsqueeze(1))
        return output, density


class Density_Softmax(nn.Module):
    '''
    返回正态分布概率密度并取对数,方便后续计算
    '''
    def __init__(self,out_features,in_features):
        super(Density_Softmax, self).__init__()
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        self.weight2 = Parameter(torch.FloatTensor(out_features, ))
        nn.init.xavier_uniform_(self.weight)
        # self.weight =softmax_head.weight
        # 一个常数
        self.bias =  torch.log(torch.tensor(2.0 * torch.pi))

    def forward(self, mu, std):
        # 优化后的运算
        # 计算mu^2/std^2
        weight = self.weight # class * dim 学习类中心
        weight2 = self.weight2 # class * 1 学习每个类的权重
        weight2_expanded = weight2.unsqueeze(0)

        temp_mu = ((mu/std)**2).sum(dim=1).unsqueeze(1) # B*1
        # 计算weight^2/std^2
        temp_weight = F.linear(torch.reciprocal(std)**2 , weight**2) # B*C
        # 计算mu*weight/std^2
        temp_mu_weight = 2 * F.linear(mu/std**2, weight) # B*C
        density = weight2 *(temp_mu + temp_weight - temp_mu_weight) + torch.log(torch.sum(std**2, dim=1)).unsqueeze(1)
        return density


class ArcFace(nn.Module):
    r"""Implement of ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            device_id: the ID of GPU where the model will be trained by model parallel.
                       if device_id=None, it will be trained on CPU without model parallel.
            s: norm of input feature
            m: margin
            cos(theta+m)
        """
    def __init__(self, in_features, out_features, device_id, s = 64.0, m = 0.50, easy_margin = False):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device_id = device_id

        self.s = s
        self.m = m

        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        if self.device_id == None:
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        else:
            x = input
            sub_weights = torch.chunk(self.weight, len(self.device_id), dim=0)
            temp_x = x.cuda(self.device_id[0])
            weight = sub_weights[0].cuda(self.device_id[0])
            cosine = F.linear(F.normalize(temp_x), F.normalize(weight))
            for i in range(1, len(self.device_id)):
                temp_x = x.cuda(self.device_id[i])
                weight = sub_weights[i].cuda(self.device_id[i])
                cosine = torch.cat((cosine, F.linear(F.normalize(temp_x), F.normalize(weight)).cuda(self.device_id[0])), dim=1)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size())
        if self.device_id != None:
            one_hot = one_hot.cuda(self.device_id[0])
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        return output


class CosFace(nn.Module):
    r"""Implement of CosFace (https://arxiv.org/pdf/1801.09414.pdf):
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        device_id: the ID of GPU where the model will be trained by model parallel.
                       if device_id=None, it will be trained on CPU without model parallel.
        s: norm of input feature
        m: margin
        cos(theta)-m
    """
    def __init__(self, in_features, out_features, device_id, s = 64.0, m = 0.35):
        super(CosFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device_id = device_id
        self.s = s
        self.m = m

        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        if self.device_id == None:
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        else:
            x = input
            sub_weights = torch.chunk(self.weight, len(self.device_id), dim=0)
            temp_x = x.cuda(self.device_id[0])
            weight = sub_weights[0].cuda(self.device_id[0])
            cosine = F.linear(F.normalize(temp_x), F.normalize(weight))
            for i in range(1, len(self.device_id)):
                temp_x = x.cuda(self.device_id[i])
                weight = sub_weights[i].cuda(self.device_id[i])
                cosine = torch.cat((cosine, F.linear(F.normalize(temp_x), F.normalize(weight)).cuda(self.device_id[0])), dim=1)
        phi = cosine - self.m
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size())
        if self.device_id != None:
            one_hot = one_hot.cuda(self.device_id[0])
        # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features = ' + str(self.in_features) \
               + ', out_features = ' + str(self.out_features) \
               + ', s = ' + str(self.s) \
               + ', m = ' + str(self.m) + ')'

class SphereFace(nn.Module):
    r"""Implement of SphereFace (https://arxiv.org/pdf/1704.08063.pdf):
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        device_id: the ID of GPU where the model will be trained by model parallel.
                       if device_id=None, it will be trained on CPU without model parallel.
        m: margin
        cos(m*theta)
    """
    def __init__(self, in_features, out_features, device_id, m = 4):
        super(SphereFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.base = 1000.0
        self.gamma = 0.12
        self.power = 1
        self.LambdaMin = 5.0
        self.iter = 0
        self.device_id = device_id

        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        # duplication formula
        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

    def forward(self, input, label):
        # lambda = max(lambda_min,base*(1+gamma*iteration)^(-power))
        self.iter += 1
        self.lamb = max(self.LambdaMin, self.base * (1 + self.gamma * self.iter) ** (-1 * self.power))

        # --------------------------- cos(theta) & phi(theta) ---------------------------
        if self.device_id == None:
            cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
        else:
            x = input
            sub_weights = torch.chunk(self.weight, len(self.device_id), dim=0)
            temp_x = x.cuda(self.device_id[0])
            weight = sub_weights[0].cuda(self.device_id[0])
            cos_theta = F.linear(F.normalize(temp_x), F.normalize(weight))
            for i in range(1, len(self.device_id)):
                temp_x = x.cuda(self.device_id[i])
                weight = sub_weights[i].cuda(self.device_id[i])
                cos_theta = torch.cat((cos_theta, F.linear(F.normalize(temp_x), F.normalize(weight)).cuda(self.device_id[0])), dim=1)

        cos_theta = cos_theta.clamp(-1, 1)
        cos_m_theta = self.mlambda[self.m](cos_theta)
        theta = cos_theta.data.acos()
        k = (self.m * theta / 3.14159265).floor()
        phi_theta = ((-1.0) ** k) * cos_m_theta - 2 * k
        NormOfFeature = torch.norm(input, 2, 1)

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cos_theta.size())
        if self.device_id != None:
            one_hot = one_hot.cuda(self.device_id[0])
        one_hot.scatter_(1, label.view(-1, 1), 1)

        # --------------------------- Calculate output ---------------------------
        output = (one_hot * (phi_theta - cos_theta) / (1 + self.lamb)) + cos_theta
        output *= NormOfFeature.view(-1, 1)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features = ' + str(self.in_features) \
               + ', out_features = ' + str(self.out_features) \
               + ', m = ' + str(self.m) + ')'


def l2_norm(input, axis = 1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

class Am_softmax(nn.Module):
    r"""Implement of Am_softmax (https://arxiv.org/pdf/1801.05599.pdf):
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        device_id: the ID of GPU where the model will be trained by model parallel.
                       if device_id=None, it will be trained on CPU without model parallel.
        m: margin
        s: scale of outputs
    """
    def __init__(self, in_features, out_features, device_id, m = 0.35, s = 30.0):
        super(Am_softmax, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.s = s
        self.device_id = device_id

        self.kernel = Parameter(torch.Tensor(in_features, out_features))
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)  # initialize kernel

    def forward(self, embbedings, label):
        if self.device_id == None:
            kernel_norm = l2_norm(self.kernel, axis = 0)
            cos_theta = torch.mm(embbedings, kernel_norm)
        else:
            x = embbedings
            sub_kernels = torch.chunk(self.kernel, len(self.device_id), dim=1)
            temp_x = x.cuda(self.device_id[0])
            kernel_norm = l2_norm(sub_kernels[0], axis = 0).cuda(self.device_id[0])
            cos_theta = torch.mm(temp_x, kernel_norm)
            for i in range(1, len(self.device_id)):
                temp_x = x.cuda(self.device_id[i])
                kernel_norm = l2_norm(sub_kernels[i], axis = 0).cuda(self.device_id[i])
                cos_theta = torch.cat((cos_theta, torch.mm(temp_x, kernel_norm).cuda(self.device_id[0])), dim=1)

        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        phi = cos_theta - self.m
        label = label.view(-1, 1)  # size=(B,1)
        index = cos_theta.data * 0.0  # size=(B,Classnum)
        index.scatter_(1, label.data.view(-1, 1), 1)
        index = index.byte()
        output = cos_theta * 1.0
        output[index] = phi[index]  # only change the correct predicted output
        output *= self.s  # scale up in order to make softmax work, first introduced in normface

        return output


if __name__ == "__main__":
    # weight = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    # weight = torch.randn(80000, 512)
    labels = torch.randn(512)
    mu_dul = torch.randn(512,512)
    std_dul = torch.randn(512,512)
    # mu_dul = torch.tensor([[3, 4, 5], [6, 7, 8], [9, 10, 11]])
    # std_dul = torch.tensor([[0.5, 0.5, 1], [1, 1, 2], [2, 2, 3]])
    HEAD = Density_Softmax(80000,512)
    # outputs是两重for循环的结果，density是优化后的结果
    density = HEAD(mu_dul, std_dul)
    print(outputs)
    print(density)