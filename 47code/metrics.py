from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
import numpy as np
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

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

    # def forward(self, x, label):
    #     if self.device_id == None:
    #         # out = F.linear(x, self.weight, self.bias)
    #         out =F.linear(x,self.weight)
    #     else:
    #         sub_weights = torch.chunk(self.weight, len(self.device_id), dim=0)
    #         sub_biases = torch.chunk(self.bias, len(self.device_id), dim=0)
    #         temp_x = x.cuda(self.device_id[0])
    #         weight = sub_weights[0].cuda(self.device_id[0])
    #         bias = sub_biases[0].cuda(self.device_id[0])
    #         # out = F.linear(temp_x, weight, bias)
    #         out = F.linear(temp_x,weight)
    #         for i in range(1, len(self.device_id)):
    #             temp_x = x.cuda(self.device_id[i])
    #             weight = sub_weights[i].cuda(self.device_id[i])
    #             bias = sub_biases[i].cuda(self.device_id[i])
    #             # out = torch.cat((out, F.linear(temp_x, weight, bias).cuda(self.device_id[0])), dim=1)
    #             out = torch.cat((out, F.linear(temp_x, weight).cuda(self.device_id[0])), dim=1)
    #     return out
    def scoring(self, embedding, sample_weight, sample_bias,nontrivial):
        # return self.head(embedding)
        return torch.matmul(embedding[:, None, :], (sample_weight * nontrivial).permute(0, 2, 1)).squeeze() + sample_bias

    def forward(self, x, mu, var,labels):
        weight = self.weight
        sample_weight = weight[labels]
        
        topk_indice = findConfounders(weight,sample_weight,K=512) # B *k
        topk_weight = weight[topk_indice]
        topk_bias = self.bias[topk_indice]
        all_class_density = torch.exp(- (topk_weight - mu[:, None, :]) ** 2 / (2 * var[:, None, :]))   # B * K * D
        confid = all_class_density / torch.clamp(all_class_density.sum(dim=1, keepdim=True), min=1e-8)
        max_confid, _ = confid.max(dim=1, keepdim=True)
        nontrivial = (confid >= torch.clamp(max_confid * 0.5, max=0.1)).detach()  # B * k * D
        score = self.scoring(x,topk_weight,topk_bias, nontrivial)
        return score,topk_indice,all_class_density,nontrivial       

    
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



import torch.nn.functional as F

def findConfounders(w, sample_w, K, method="euc"):
    if method == "euc": # Euclidean distance:  (W[None, :, :] - W[y][:, None, :])^2  : B * C * D
       sample_w2 = (sample_w ** 2).sum(dim=1, keepdim=True)  # B * 1
       w2 = (w ** 2).sum(dim=1)[None, :]                     # 1 * C
       ww = 2 * torch.matmul(sample_w, w.T)                  # B * C
       
       dis = sample_w2 - ww + w2                             # B * C
       
       _, indices = dis.topk(K, dim=1,largest = False)                       # topk -> B * K
       
    else:# Cosine similarity:
       dis = F.cosine_similarity(sample_w[:, None, :], w[None, :, :], dim=-1)     # B * C
       _, indices = dis.topk(K, dim=1, largest=True)                              # topk -> B * K  
    
    return indices

class Density_Softmax(nn.Module):
    '''
    用分块循环的方式计算按类别求和结果 B*C*D -> B*D
    '''
    def __init__(self, in_features, out_features):
        super(Density_Softmax, self).__init__()
        self.topk = True

    # def forward(self, center, mu, var, labels): # center: c *dim
        
    #     batch_size = mu.shape[0]
    #     class_num = center.shape[0]
    #     dim_size = center.shape[1]
    #     weight = center
    #     # weight = abs(center).detach()           # weight 复用center，可以选择detach # c *dim
    #     select_weight = weight[labels]          # B * dim
    #     dis = (select_weight - mu) ** 2 / (2 * var)# B * dim
    #     density = abs(select_weight) * torch.exp(-dis)  # B * dim
    #     # nonzero_ratio = ((torch.sum(select_weight != 0,dim=1))).unsqueeze(1) # B 计算非零元素个数
    #     nonzeros = (select_weight != 0).sum(dim=-1, keepdim=True) # B 计算非零元素个数
    #     mu = mu.unsqueeze(1)                      # B * 1 * dim
    #     var = var.unsqueeze(1)                    # B * 1 * dim

    #     if not self.topk:
    #     # 窗口循环计算—— 所有类别求和版本// 计算图未发生变化，显存占用不变
    #         all_class_density = torch.zeros(batch_size, dim_size).to(center.device) # B * D 
    #         chunk_size = 2                     # 每次计算的块大小
    #         n_chunks = class_num // chunk_size # 分块数
    #         weight_chunks = torch.chunk(weight, n_chunks, dim=0) # 分块
            
    #         for weight_chunk in weight_chunks: # 按块计算
    #             cur_weight_chunk = weight_chunk.unsqueeze(0)                        # 1 * chunk_size * D
    #             temp_result = ((cur_weight_chunk - mu) ** 2 ).sum(dim=1) /(2 * var) # B * D
    #             all_class_density += temp_result 

    #         output = (density / (all_class_density + 1e-8) / nonzero_ratio) # B * dim  -> P
    #         output = torch.log(output + 1e-8)                           # B * dim -> logP  
    #         return output

    #     # 窗口循环计算—— 选择topk类别求和版本
    #     else:
    #         K = 512
    #         # select_weight_expand = select_weight.unsqueeze(1) # B * 1 *D
    #         # weight_expand = weight.unsqueeze(0)               # 1 * C *D
    #         # 对dim维度进行求和
    #         indices = findConfounders(weight, select_weight, K)
    #         # select_weight_term = (select_weight ** 2).sum(dim=1).unsqueeze(dim=1)     # B * D -> B * 1
    #         # weight_term = (weight ** 2).sum(dim=1).unsqueeze(dim=0)                   # C * D -> 1 * C
    #         # weight_select_weight = F.linear(select_weight, weight)                    # B * C
    #         # weight_dis = select_weight_term - 2 * weight_select_weight + weight_term  # B * C
    #         # _, indices = weight_dis.topk(K, dim=1, largest=True)                      # topk -> B * K
    #         topk_weight = weight[indices.view(-1)].view(batch_size, K, dim_size)        # BK * D -> B* K * D

    #         dis = ((topk_weight - mu) ** 2) /(2 * var)                                  # B *K *D
    #         topk_density = torch.clamp(torch.exp(-dis).sum(dim=1), min=1e-8)            # B *D
            
    #         output = (density / topk_density).sum(dim=-1) / torch.clamp(nonzeros, min=1)# B *D  -> P
    #         output = torch.log(torch.clamp(output.mean(), min=1e-8))                    # B *D  -> logP  

    #         return output
    def forward(self, weight, mu, var, labels,all_class_density, nontrivial):
        sample_weight = weight[labels]                                         # B * dim
        density = torch.exp(- (sample_weight - mu).detach() ** 2 / (2 * var))  # B * dim
        # all_class_density = torch.exp(- (sample_weight[None, :, :] - mu[:, None, :]) ** 2 / (2 * var[:, None, :]))   # B * K * D
        total_density = torch.clamp(all_class_density.sum(dim=1), min=1e-8)    # B * D ,在类别维度进行求和
        # nonzeros = (sample_weight != 0).sum(dim=-1, keepdim=True) # B 计算非零元素个数
        
        # print(torch.isnan(total_density).any())
 
        method = 1 
        if method == 0:
            _, indices = all_class_density.topk(2, dim=1, largest=True, sorted=True)                      # B * 2 * D

            overly = (((indices[:, 0] == labels[:, None]) & (density - all_class_density.gather(1, indices[:, 1][:, None, :]).squeeze() >= 0.2 * total_density))).int()
        else:
            _, indices = all_class_density.min(dim=1)
            nontrivial = nontrivial.scatter(1, indices[:, None, :], False)                     #b*k*d

            minv = (all_class_density + (~ nontrivial) * 1000).min(dim=1, keepdim=True).values # b * d
            maxv = (all_class_density - nontrivial * 1000).max(dim=1, keepdim=True).values
            # print(labels[:, None, None].expand(-1, -1, mu.shape[-1]))
            # print(minv - maxv)
            # overly = ((nontrivial.gather(1, labels[:, None, None].expand(-1, -1, mu.shape[-1]))) & \
            #           ((minv - maxv) >= 0.2 * total_density[:,None,:])).int() # label 超出了nontrivial的范围
            # overly = ((nontrivial.gather(2, labels[:, None, None].expand(-1, -1, mu.shape[-1]))) & ((minv - maxv) >= 0.2 * total_density[:,None,:])).int()
            overly = ((nontrivial) & ((minv - maxv) >= 0.2 * total_density[:,None,:])).int()
            # overly = (((nontrivial[labels[:, None,None]])) & ((minv - maxv) >= 0.2 * total_density[:,None,:])).int()
        # print(torch.isnan(overly).any(earl)
        confid = density[:,None,:] / total_density[:,None,:] * (1 - overly)
        penalize = True 
        if penalize:
            density_detached = torch.exp(- (sample_weight - mu).detach() ** 2 / (2 * var))
            topkindice = findConfounders(weight,sample_weight,512)
            topk_weight = weight[topkindice]
            all_class_density_detached = torch.exp(- (topk_weight - mu[:, None, :]).detach() ** 2 / (2 * var[:, None, :]))
     
            total_density_detached = torch.clamp(all_class_density_detached.sum(dim=1), min=1e-8)        #B * D

            confid = confid - density_detached[:,None,:] / total_density_detached[:,None,:] * overly

        # confid = torch.clamp(confid.sum(dim=-1), min=1e-8) / torch.clamp(nonzeros, min=1)            #B
        confid = confid.mean(dim=-1)            #B
        assert(not torch.any(torch.isnan(confid))),"c "
        return confid.mean()
# class Density_Softmax(nn.Module):
#     '''
#     返回正态分布概率密度并取对数,方便后续计算 0612版本
#     '''
#     def __init__(self,out_features,in_features):
#         super(Density_Softmax, self).__init__()
#         # self.weight =softmax_head.weight
#         self.center = Parameter(torch.FloatTensor(out_features, in_features))
#         self.weight = Parameter(torch.FloatTensor(out_features, in_features)) 
#         nn.init.xavier_uniform_(self.center)
#         # nn.init.uniform_(self.weight, 1e-8, 1)
#         nn.init.uniform_(self.weight, 1e-5, 1)
#         self.bias = torch.log(torch.tensor(2.0 * torch.pi))

#     def forward(self, mu, var):
#         # 优化后的运算
#         # 计算mu^2/std^2
#         center = self.center
#         weight = self.weight
#         temp_mu_clamped = torch.exp(((mu **2/var)).clamp(max= 10)) # B*dim
#         temp_mu = torch.log(torch.clamp(F.linear(temp_mu_clamped, weight),min=1e-8)) # B*C
#         # 计算weight^2/std^2
#         temp_weight = F.linear(torch.reciprocal(var) , center**2)
#         # 计算mu*weight/std^2
#         temp_mu_weight = 2 * F.linear(mu/var, center)
#         density = -0.5*((temp_mu + temp_weight - temp_mu_weight) + torch.sum(torch.log(var), dim=1).unsqueeze(1)) + self.bias
#         # print(torch.isnan(density).any())
#         return density
    
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
    # feat = F.normalize(torch.rand(256, 64, requires_grad=True))
    # lbl = torch.randint(high=10, size=(256,))

    # inp_sp, inp_sn = convert_label_to_similarity(feat, lbl)

    # criterion = CircleLoss(m=0.25, gamma=256)
    # circle_loss = criterion(inp_sp, inp_sn)

    # print(circle_loss)
    head = Density_Softmax(512, 10) # 512维特征，10个类别
    softmax_head = Softmax(512, 10, None) # 10个类别，512维dim
    mu = torch.rand(256, 512) # B *D 256个样本，512个dim
    std = torch.rand(256, 512)# B *D 256个样本，512个dim
    labels = torch.randint(high=10, size=(256,))
    output = head(softmax_head.weight,mu, std,labels)
    print(output)
