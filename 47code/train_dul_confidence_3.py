import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:2048"
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segment:true"
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from config import Backbone_Dict, dul_args_func
from head.metrics import ArcFace, CosFace, SphereFace, Am_softmax, Softmax,Density_Softmax
from loss.focal import FocalLoss
from loss.density_loss import Density_loss
from util.utils import make_weights_for_balanced_classes, separate_irse_bn_paras, \
                       warm_up_lr, schedule_lr, get_time, AverageMeter, accuracy, add_gaussian_noise

from tensorboardX import SummaryWriter, writer
import os
import time
# import numpy as np
# from PIL import Image
# import random


class DUL_Trainer():
    def __init__(self, dul_args):
        self.dul_args = dul_args
        # self.dul_args.model_save_folder = f"./checkpoints/density_{dul_args.kl_scale}_{dul_args.var_scale}_{dul_args.conf_scale}/"
        self.dul_args.gpu_id = [int(item) for item in self.dul_args.gpu_id]
        self.dul_args.stages = [int(item) for item in self.dul_args.stages]

    def _report_configurations(self):
        print('=' * 60)
        print('Experiment time: ', get_time())
        print('=' * 60)
        print('Overall Configurations:')
        print('=' * 60)
        for k in self.dul_args.__dict__:
            print(" '{}' : '{}' ".format(k, str(self.dul_args.__dict__[k])))
        os.makedirs(self.dul_args.model_save_folder, exist_ok=True)
        os.makedirs(self.dul_args.log_tensorboard, exist_ok=True)
        writer = SummaryWriter(self.dul_args.log_tensorboard)
        return writer


    def _data_loader(self):
        if self.dul_args.center_crop:
            train_transform = transforms.Compose([ 
            transforms.Resize([int(128 * self.dul_args.input_size[0] / 112), int(128 * self.dul_args.input_size[0] / 112)]),
            transforms.RandomCrop([self.dul_args.input_size[0], self.dul_args.input_size[1]]),
            transforms.RandomHorizontalFlip(),
            add_gaussian_noise(p=self.dul_args.image_noise),
            transforms.ToTensor(),
            transforms.Normalize(mean = self.dul_args.rgb_mean,
                                 std = self.dul_args.rgb_std),
            #transforms.RandomErasing(scale=(0.02,0.25))
        ])
        else:
            train_transform = transforms.Compose([ # refer to https://pytorch.org/docs/stable/torchvision/transforms.html for more build-in online data augmentation
            transforms.Resize([112, 112]), # smaller side resized
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean = self.dul_args.rgb_mean,
                                 std = self.dul_args.rgb_std),
            transforms.RandomErasing(scale=(0.02,0.25))
        ])

        dataset_train = datasets.ImageFolder(self.dul_args.trainset_folder, train_transform)

        # ----- create a weighted random sampler to process imbalanced data
        weights = make_weights_for_balanced_classes(dataset_train.imgs, len(dataset_train.classes))
        weights = torch.DoubleTensor(weights)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

        train_loader = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler, batch_size=self.dul_args.batch_size,
            pin_memory=self.dul_args.pin_memory, num_workers=self.dul_args.num_workers,
            drop_last=self.dul_args.drop_last,
        )

        num_class = len(train_loader.dataset.classes)
        print('=' * 60)
        print("Number of Training Classes: '{}' ".format(num_class))

        print('=' * 60)
        print("Number of Training Images: '{}' ".format(len(train_loader.dataset)))

        return train_loader, num_class


    def _model_loader(self, num_class):
        # ----- backbone generate
        BACKBONE = Backbone_Dict[self.dul_args.backbone_name]
        print("=" * 60)
        print("Backbone Generated: '{}' ".format(self.dul_args.backbone_name))

        # ----- head generate
        Head_Dict = {
            'ArcFace': ArcFace(in_features = self.dul_args.embedding_size, out_features = num_class, device_id = self.dul_args.gpu_id, s=self.dul_args.arcface_scale),
            'CosFace': CosFace(in_features = self.dul_args.embedding_size, out_features = num_class, device_id = self.dul_args.gpu_id),
            'SphereFace': SphereFace(in_features = self.dul_args.embedding_size, out_features = num_class, device_id = self.dul_args.gpu_id),
            'Am_softmax': Am_softmax(in_features = self.dul_args.embedding_size, out_features = num_class, device_id = self.dul_args.gpu_id),
            'Softmax': Softmax(in_features = self.dul_args.embedding_size, out_features = num_class, device_id = self.dul_args.gpu_id),
            # 'Density':Density_Softmax(Softmax(in_features = self.dul_args.embedding_size, out_features = num_class, device_id = self.dul_args.gpu_id))
            'Density':Density_Softmax(in_features = self.dul_args.embedding_size, out_features = num_class)
        }
        HEAD = Head_Dict[self.dul_args.head_name]
        SOFTMAX_HEAD = Head_Dict['Softmax']
        # SOFTMAX_HEAD = Head_Dict['ArcFace']
        print("=" * 60)
        print("Head Generated: '{}' ".format(self.dul_args.head_name))

        # ----- loss generate
        Loss_Dict = {
            'Focal': FocalLoss(),
            'Softmax': nn.CrossEntropyLoss(),
            'Density':Density_loss(),
            'NLLLoss': nn.NLLLoss()
        }
        LOSS = Loss_Dict[self.dul_args.loss_name]
        print("=" * 60)
        print("Loss Generated: '{}' ".format(self.dul_args.loss_name))
        # ----- separate batch_norm parameters from others; do not do weight decay for batch_norm parameters to improve the generalizability
        backbone_paras_only_bn, backbone_paras_wo_bn = separate_irse_bn_paras(BACKBONE)
        _, head_paras_wo_bn = separate_irse_bn_paras(HEAD)
        _, softmax_head_paras_wo_bn = separate_irse_bn_paras(SOFTMAX_HEAD)

        # ----- optimizer generate
        Optimizer_Dict = {
            'SGD': optim.SGD([{'params': backbone_paras_wo_bn + head_paras_wo_bn + softmax_head_paras_wo_bn, 'weight_decay': self.dul_args.weight_decay}, 
                            {'params': backbone_paras_only_bn}], lr=self.dul_args.lr, momentum=self.dul_args.momentum),
            'Adam': optim.Adam([{'params': backbone_paras_wo_bn + head_paras_wo_bn, 'weight_decay': self.dul_args.weight_decay}, 
                            {'params': backbone_paras_only_bn}], lr=self.dul_args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
        }
        OPTIMIZER = Optimizer_Dict[self.dul_args.optimizer]
        print("=" * 60)
        print("Optimizer Generated: '{}' ".format(self.dul_args.optimizer))
        print(OPTIMIZER)

        # ----- optional resume
        if self.dul_args.resume_backbone or self.dul_args.resume_head:
            print("=" * 60)
            if os.path.isfile(self.dul_args.resume_backbone):
                print("Loading Backbone Checkpoint '{}'".format(self.dul_args.resume_backbone))
                BACKBONE.load_state_dict(torch.load(self.dul_args.resume_backbone))
            if os.path.isfile(self.dul_args.resume_head):
                print("Loading Head Checkpoint '{}'".format(self.dul_args.resume_head))
                try:
                    HEAD.load_state_dict(torch.load(self.dul_args.resume_head))
                except Exception as e:
                    print(e)
        else:
            print("No Checkpoint Found at '{}' and '{}'. Please Have a Check or Continue to Train from Scratch".\
                format(self.dul_args.resume_backbone, self.dul_args.resume_head))

        # ----- multi-gpu or single-gpu
        if self.dul_args.multi_gpu:
            BACKBONE = nn.DataParallel(BACKBONE, device_ids=self.dul_args.gpu_id).cuda()
            SOFTMAX_HEAD =SOFTMAX_HEAD.cuda()
            HEAD = HEAD.cuda()
            LOSS = LOSS.cuda()
        else:
            BACKBONE = BACKBONE.cuda()
            SOFTMAX_HEAD =SOFTMAX_HEAD.cuda()
            HEAD = HEAD.cuda()
            LOSS = LOSS.cuda()

        return BACKBONE, HEAD, SOFTMAX_HEAD, LOSS, OPTIMIZER



    def _dul_runner(self):
        writer = self._report_configurations()

        train_loader, num_class = self._data_loader()

        BACKBONE, HEAD,SOFTMAX_HEAD, LOSS, OPTIMIZER = self._model_loader(num_class=num_class)

        DISP_FREQ = len(train_loader) // 100 # frequency to display training loss & acc

        NUM_EPOCH_WARM_UP = self.dul_args.warm_up_epoch
        NUM_BATCH_WARM_UP = int(len(train_loader) * NUM_EPOCH_WARM_UP)
        batch = 0  # batch index

        print('=' * 60)
        print("Display Freqency: '{}' ".format(DISP_FREQ))
        print("Number of Epoch for Warm Up: '{}' ".format(NUM_EPOCH_WARM_UP))
        print("Number of Batch for Warm Up: '{}' ".format(NUM_BATCH_WARM_UP))
        print('Start Training: ')

        for epoch in range(self.dul_args.num_epoch):
            if epoch == self.dul_args.stages[0]:
                schedule_lr(OPTIMIZER)
            elif epoch == self.dul_args.stages[1]:
                schedule_lr(OPTIMIZER)
            if epoch < self.dul_args.resume_epoch:
                continue
            
            BACKBONE.train()  # set to training mode
            HEAD.train()
            SOFTMAX_HEAD.train()

            BACKBONE.training = True

            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            losses_KL = AverageMeter()
            # losses_confidence = AverageMeter()
            # losses_cls = AverageMeter()
            # losses_var = AverageMeter()

            for inputs, labels in train_loader:
                if (epoch + 1 <= NUM_EPOCH_WARM_UP) and (batch + 1 <= NUM_BATCH_WARM_UP): # adjust LR for each training batch during warm up
                    warm_up_lr(batch + 1, NUM_BATCH_WARM_UP, self.dul_args.lr, OPTIMIZER)
                
                inputs = inputs.cuda()
                labels = labels.cuda().long()
                loss = 0
                
                mu_dul,var_dul = BACKBONE(inputs) 
                std_dul = torch.sqrt(var_dul)
                epsilon = torch.randn_like(std_dul)
                features = mu_dul + epsilon * std_dul
                # Softmax_outputs = SOFTMAX_HEAD(features, labels)
                Softmax_outputs,topk_indice ,all_class_density,nontrivial= SOFTMAX_HEAD(features, mu_dul, var_dul, labels) # B*K,
                # print(Softmax_outputs.min())
                # print(Softmax_outputs.mean())
                loss_kl = ((var_dul + mu_dul ** 2 - torch.log(var_dul) - 1) * 0.5).mean() # KL损失 此处考虑添加w
                losses_KL.update(loss_kl.item(), inputs.size(0))
                # loss += self.dul_args.kl_scale * loss_kl        #超参默认=0.01
                # if torch.isnan(loss_kl):
                #     print("loss_kl is nan ")
                # SOFTMAX_HEAD :Softmax 或arcface 分类头
                # HEAD :计算混合高斯分布概率结果-> B
                minv_softmax = Softmax_outputs.min().item()
                outputs =HEAD(SOFTMAX_HEAD.weight,mu_dul,var_dul,labels,all_class_density,nontrivial) # 计算混合高斯分布概率在对应类别的结果 ->B
                loss_confidence = outputs.mean()                         # 置信度损失
                Softmax_outputs_bc = torch.full((mu_dul.shape[0],num_class),-1e8).type(Softmax_outputs.type()).cuda()
                # 创建一个索引张量，用于指定 small_tensor 中的元素应该被填充到 large_tensor 中的哪些位置
                # indices = torch.arange(Softmax_outputs.shape[1]).unsqueeze(0).repeat(mu_dul.shape[0], 1).cuda()
                Softmax_outputs_bc.scatter_(1,topk_indice,Softmax_outputs)
                # print(Softmax_outputs_bc[torch.arange(0,256), labels])
                loss_cls = LOSS(Softmax_outputs_bc,labels)          # 分类头Softmax/ArcFace的分类交叉熵损失
                loss -= self.dul_args.conf_scale *loss_confidence # 置信度损失 超参系数=0.5
                loss += loss_cls                                 # 分类损失 未添加超参即系数为1.0
                loss -= self.dul_args.var_scale * var_dul.mean() # 添加L2正则项，让方差尽可能大 超参系数=0.5
                # loss_copy = loss.clone().detach()

                if torch.isnan(loss):
                    print("loss is nan,save tensor")
                    # torch.save(outputs,'outputs.pt')
                    # torch.save(HEAD.center, 'center.pt')
                    # torch.save(HEAD.weight,'weight.pt')
                    # torch.save(mu_dul, 'mu_dul.pt')
                    # torch.save(var_dul, 'val_dul.pt')
                    return

                # measure accuracy and record loss
                prec1, prec5 = accuracy(Softmax_outputs.data, labels, topk = (1, 5))
                losses.update(loss_cls.data.item(), inputs.size(0))
                top1.update(prec1.data.item(), inputs.size(0))
                top5.update(prec5.data.item(), inputs.size(0))
                # losses_confidence.update(loss_confidence.data.item(), inputs.size(0))
                # losses_cls.update(loss_cls.item(), inputs.size(0))
                # losses_var.update(var_dul.mean().item(), inputs.size(0))


                # compute gradient and do SGD step
                OPTIMIZER.zero_grad()
                loss.backward()
                OPTIMIZER.step()

                # 使用tensorboard记录特定类别的variance和weight
                # selected_classes = range(5)
                # for c in selected_classes:
                #     mask = labels == c
                #     var_c = var_dul[mask]
                #     # 将相对索引转为绝对索引
                #     indices = labels[mask].long()
                #     w_c = HEAD.weight[indices]
                #     if var_c.numel() == 0:
                #         continue
                #     writer.add_scalar("var_{}".format(c),var_c.mean(),global_step=batch)
                #     writer.add_scalar("weight_{}".format(c),w_c.mean(),global_step=batch)

                # dispaly training loss & acc every DISP_FREQ
                if ((batch + 1) % DISP_FREQ == 0) and batch != 0:
                    print("=" * 60, flush=True)
                    print('Epoch {}/{} Batch {}/{}\t'
                          'Time {}\t'
                          'Training Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Training Loss_KL {loss_KL.val:.4f} ({loss_KL.avg:.4f})\t'
                        #   'Training Loss_Confidence {loss_confidence.val:.4f} ({loss_confidence.avg:.4f})\t'
                        #   'Training Loss_Cls {loss_cls.val:.4f} ({loss_cls.avg:.4f})\t'
                        #   'Training Loss_Var {loss_var.val:.4f} ({loss_var.avg:.4f})\t'
                          'Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Training Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        epoch + 1, self.dul_args.num_epoch, batch + 1, len(train_loader) * self.dul_args.num_epoch, time.asctime(time.localtime(time.time())), loss = losses, loss_KL=losses_KL,  top1 = top1, top5 = top5), flush=True)

                batch += 1 # batch index
            # training statistics per epoch (buffer for visualization)
            epoch_loss = losses.avg
            # epoch_loss_KL = losses_KL.avg
            # epoch_loss_confidence = losses_confidence.avg
            # epoch_loss_cls = losses_cls.avg
            epoch_acc = top1.avg
            writer.add_scalar("Training_Loss", epoch_loss, epoch + 1)
            writer.add_scalar("Training_Accuracy", epoch_acc, epoch + 1)
            # writer.add_scalar("Training_Loss_KL", epoch_loss_KL, epoch + 1)
            # writer.add_scalar("Training_Loss_Confidence", epoch_loss_confidence, epoch + 1)
            # writer.add_scalar("Training_Loss_Cls", epoch_loss_cls, epoch + 1)
            # writer.add_scalar("Training_Loss_Var", losses_var.avg, epoch + 1)
            print("=" * 60, flush=True)
            print('Epoch: {}/{}\t'
                  'Training Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Training Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
            epoch + 1, self.dul_args.num_epoch, loss = losses, top1 = top1, top5 = top5), flush=True)

            # ----- save model
            if epoch>10:
                print("=" * 60, flush=True)
                print('Saving NO.EPOCH {} trained model'.format(epoch+1), flush=True)
                if self.dul_args.multi_gpu:
                    # if not os.path.exists(self.dul_args.model_save_folder):
                    #     os.makedirs(self.dul_args.model_save_folder)
                    torch.save(BACKBONE.module.state_dict(), os.path.join(self.dul_args.model_save_folder, "Backbone_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(self.dul_args.backbone_name, epoch + 1, batch, get_time())))
                    torch.save(HEAD.state_dict(), os.path.join(self.dul_args.model_save_folder, "Head_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(self.dul_args.head_name, epoch + 1, batch, get_time())))
                else:
                    torch.save(BACKBONE.state_dict(), os.path.join(self.dul_args.model_save_folder, "Backbone_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(self.dul_args.backbone_name, epoch + 1, batch, get_time())))
                    torch.save(HEAD.state_dict(), os.path.join(self.dul_args.model_save_folder, "Head_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(self.dul_args.head_name, epoch + 1, batch, get_time())))
        print('=' * 60, flush=True)
        print('Training process finished!', flush=True)
        print('=' * 60, flush=True)


if __name__ == '__main__':
    dul_train = DUL_Trainer(dul_args_func())
    dul_train._dul_runner()
