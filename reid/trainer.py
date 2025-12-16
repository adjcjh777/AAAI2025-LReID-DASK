from __future__ import print_function, absolute_import
import time

from torch.nn import functional as F
import torch
import torch.nn as nn
from .utils.meters import AverageMeter
from .utils.feature_tools import *

from reid.utils.make_loss import make_loss
import copy
from reid.utils.color_transformer import ColorTransformer
from reid.metric_learning.distance import cosine_similarity
import random
import cv2
import os

import wandb

def remap(inputs_r,imgs_origin, training_phase, save_dir,dataset_name):
    mean=torch.tensor([0.485, 0.456, 0.406])
    std=torch.tensor([0.229, 0.224, 0.225])
    x=inputs_r.detach()
    x=x.cpu()
    # print(x.shape)
    x=(x)*std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)+mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    # print(x.min(),x.max())

    x=(x*255).clamp(min=0,max=255)
    x=x.permute(0,2,3,1)
    x=x.numpy().astype('uint8')
    vis_dir=save_dir+f'/vis/{dataset_name}/'+str(training_phase)+'/'
    os.makedirs(vis_dir, exist_ok=True)


    imgs_origin=imgs_origin.permute(0,2,3,1)
    imgs_origin=(imgs_origin*255).cpu().numpy().astype('uint8')
    

    for i in range(len(x)):
        cv2.imwrite(vis_dir+f'{i}_reconstruct.png',x[i][:,:,::-1])
        cv2.imwrite(vis_dir+f'{i}_rorigin.png',imgs_origin[i][:,:,::-1])
    print("saved images in ", vis_dir)
    
class Trainer(object):
    def __init__(self,cfg,args, model, num_classes, writer=None, grad_mask=None):
        super(Trainer, self).__init__()
        self.cfg = cfg
        self.args = args
        self.model = model
        self.writer = writer
        self.AF_weight = args.AF_weight
        self.grad_mask = grad_mask

        self.loss_fn, center_criterion = make_loss(cfg, num_classes=num_classes)
        self.loss_ce=nn.CrossEntropyLoss(reduction='batchmean')
        self.KLDivLoss = nn.KLDivLoss( reduction = "batchmean")
        self.MSE=torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
        self.MAE = torch.nn.L1Loss(size_average=None, reduce=None, reduction='mean')


    def train(self, epoch, data_loader_train,  optimizer, training_phase,
              train_iters=200, add_num=0, old_model=None,rehearser=None ):

        self.model.train()
        # freeze the bn layer totally
        for m in self.model.module.base.modules():
            if isinstance(m, nn.BatchNorm2d):
                if m.weight.requires_grad == False and m.bias.requires_grad == False:
                    m.eval()
        
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_ce = AverageMeter()
        losses_tr = AverageMeter()
        losses_AF = AverageMeter()

        end = time.time()

        total_weight=[1.0,self.args.aux_weight] 
        for i in range(train_iters):
            train_inputs = data_loader_train.next()
            data_time.update(time.time() - end)

            s_inputs,imgs_origin, targets, cids, domains, = self._parse_data(train_inputs)

            if training_phase>1:
                with torch.no_grad():
                    kernel = rehearser(imgs_origin) 
                inputs_r=decode_transfer_img(self.args,imgs_origin,kernel)               
                
                datas=[
                    [s_inputs,imgs_origin, targets, cids, domains],
                    [inputs_r,imgs_origin, targets, cids, domains]
                ]            
            else:
                datas=[
                    [s_inputs,imgs_origin, targets, cids, domains]
                ]

            loss=0    
            for idx, (s_inputs,imgs_origin, targets, cids, domains) in enumerate(datas):
                targets_origin=targets
                targets =targets+ add_num                

                s_features, bn_feat, cls_outputs, feat_final_layer = self.model(s_inputs)

                '''calculate the base loss'''
                loss_ce, loss_tp = self.loss_fn(cls_outputs, s_features, targets, target_cam=None)
                  
                losses_ce.update(loss_ce.mean().item())
                losses_tr.update(loss_tp.item())


                divergence=0.

                if old_model is not None:
                    with torch.no_grad():
                        s_features_old, bn_feat_old, cls_outputs_old, feat_final_layer_old = old_model(s_inputs, get_all_feat=True)
                    if isinstance(s_features_old, tuple):
                        s_features_old=s_features_old[0]                                    
                    
                    Affinity_matrix_new = self.get_normal_affinity(s_features)  #
                    Affinity_matrix_old = self.get_normal_affinity(s_features_old)
                    # print(Affinity_matrix_new[0].cpu().tolist())
                    divergence = self.cal_KL_old(Affinity_matrix_new, Affinity_matrix_old)
                    divergence = divergence * self.AF_weight
                    
                    losses_AF.update(divergence.item())
                    
                    if self.args.l2sp_weight > 0:
                        loss_l2sp = self.cal_l2sp_loss(self.model, old_model) * self.args.l2sp_weight
                        loss += loss_l2sp
                        if self.writer != None:
                            self.writer.add_scalar(tag="loss/Loss_l2sp_{}".format(training_phase), scalar_value=loss_l2sp.item(),
                                global_step=epoch * train_iters + i)

                loss = loss+ (loss_ce + loss_tp+divergence)*total_weight[idx]
            loss=loss
                
            optimizer.zero_grad()
            loss.backward()
            
            # Apply Gradient Mask (Fisher Freezing)
            if self.grad_mask is not None:
                for n, p in self.model.named_parameters():
                    if n in self.grad_mask:
                        if p.grad is not None:
                            p.grad.data.mul_(self.grad_mask[n])

            optimizer.step()           

            batch_time.update(time.time() - end)
            end = time.time()
            if self.writer != None :
                self.writer.add_scalar(tag="loss/Loss_ce_{}".format(training_phase), scalar_value=losses_ce.val,
                        global_step=epoch * train_iters + i)
                self.writer.add_scalar(tag="loss/Loss_tr_{}".format(training_phase), scalar_value=losses_tr.val,
                        global_step=epoch * train_iters + i)

                self.writer.add_scalar(tag="time/Time_{}".format(training_phase), scalar_value=batch_time.val,
                        global_step=epoch * train_iters + i)
            
            # wandb logging
            log_dict = {
                "loss/ce": losses_ce.val,
                "loss/tr": losses_tr.val,
                "loss/af": losses_AF.val,
                "time/batch_time": batch_time.val,
                "lr": optimizer.param_groups[0]['lr'],
                "epoch": epoch,
                "iter": i,
                "training_phase": training_phase
            }
            if 'loss_l2sp' in locals():
                 log_dict["loss/l2sp"] = loss_l2sp.item()
                 
            wandb.log(log_dict)

            if (i + 1) == train_iters:
            #if 1 :
                print('Epoch: [{}][{}/{}]\t'
                    'Time {:.3f} ({:.3f})\t'
                    'Loss_ce {:.3f} ({:.3f})\t'
                    'Loss_tp {:.3f} ({:.3f})\t'
                    .format(epoch, i + 1, train_iters,
                            batch_time.val, batch_time.avg,
                            losses_ce.val, losses_ce.avg,
                            losses_tr.val, losses_tr.avg,
                ))       

    def get_normal_affinity(self,x,Norm=0.1):
        pre_matrix_origin=cosine_similarity(x,x)
        pre_affinity_matrix=F.softmax(pre_matrix_origin/Norm, dim=1)
        return pre_affinity_matrix
    def _parse_data(self, inputs):
        imgs, imgs_origin,_, pids, cids, domains = inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        imgs_origin=imgs_origin.cuda()
        return inputs,imgs_origin, targets, cids, domains
  
    def cal_KL_old(self,Affinity_matrix_new, Affinity_matrix_old):       
        Target=Affinity_matrix_old
        Affinity_matrix_new_log = torch.log(Affinity_matrix_new)
        divergence=self.KLDivLoss(Affinity_matrix_new_log, Target)  # 128*128                  
        return divergence     

    def cal_l2sp_loss(self, model, old_model):
        loss_l2sp = 0.0
        # Iterate over named parameters to match them
        # Note: model and old_model might be DataParallel wrappers
        if hasattr(model, 'module'):
            model_base = model.module.base
        else:
            model_base = model.base
            
        if hasattr(old_model, 'module'):
            old_model_base = old_model.module.base
        else:
            old_model_base = old_model.base

        # Only regularize base parameters (backbone), as classifier shape changes
        for (name, param), (old_name, old_param) in zip(model_base.named_parameters(), old_model_base.named_parameters()):
            if name != old_name:
                continue # Should not happen if models same architecture
            if param.requires_grad:
                loss_l2sp += torch.sum((param - old_param.detach()) ** 2)
        
        return 0.5 * loss_l2sp     

class RehearserTrainer(object):
    def __init__(self,args=None, rehearser=None):
        super(RehearserTrainer, self).__init__()
        self.color_transformer = ColorTransformer()
        self.train_transformer = T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.rehearser = rehearser
        self.args =args

        self.MAE = torch.nn.L1Loss(size_average=None, reduce=None, reduction='mean')

        self.blur = T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))

    
    def train(self, epoch, loader_source, train_iters=200,dataset_name=None,
              print_freq=100, writer=None):
        losses_cond = AverageMeter()

        for it in range(train_iters):
            _,s_inputs_o, _ = self._parse_data(
            loader_source.next()
            )

            trans_inputs = self.color_transformer.color_transfer_resample(s_inputs_o, lab=False)    # distribution augmentation
            if self.args.blur and random.random()>0.5: # random blur
                trans_inputs=self.blur(trans_inputs)
            kernel = self.rehearser(trans_inputs)    # learn the convolution kernel
            inputs_r=decode_transfer_img(self.args,trans_inputs,kernel)
            # # conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
            # k_w=kernel[:,:3*3*3*3]
            # BS=kernel.size(0)
            # k_w=k_w.reshape(BS,3,3,3,3)
            # k_b=kernel[:,3*3*3*3:]
            # inputs_r=torch.cat([F.conv2d(img.unsqueeze(0), weight=w, bias=b, stride=1, padding=1) for w,b,img in zip(k_w,k_b,trans_inputs)])
            target_inputs=self.train_transformer(s_inputs_o)    
            loss_c=self.MAE(inputs_r, target_inputs)
            
            losses_cond.update(loss_c.item())

            self.rehearser.optim.zero_grad()
            loss_c.backward()
            self.rehearser.optim.step()

            if (it + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\tLoss_cond {:.3f} ({:.3f})'.format(
                epoch, it + 1, train_iters,
                losses_cond.val, losses_cond.avg))
                if it<print_freq:
                    remap(inputs_r,s_inputs_o, epoch, self.args.logs_dir, dataset_name)

    def _parse_data(self, inputs):
        imgs,imgs_o, _, pids, _, _ = inputs
        # print(inputs)
        imgs_o = imgs_o.cuda()
        imgs = imgs.cuda()
        pids = pids.cuda() 
        return imgs,imgs_o, pids
# decode the reconstructed images according to the predicted instance-specific kernels
def decode_transfer_img(args,imgs,kernels):
    BS=imgs.size(0)
    for i in range(args.n_kernel):
        if args.groups==1:
            offset=3*3*3*3+3
            k_w=kernels[:,offset*i:offset*(i+1)-3]
            k_w=k_w.reshape(BS,3,3,3,3)
            k_b=kernels[:,offset*(i+1)-3:offset*(i+1)]
            imgs=torch.cat([F.conv2d(img.unsqueeze(0), weight=w, bias=b, stride=1, padding=1) for w,b,img in zip(k_w,k_b,imgs)])
        elif args.groups==3:
            offset=3*3*3+3
            k_w=kernels[:,offset*i:offset*(i+1)-3]
            k_w=k_w.reshape(BS,3,1,3,3)
            k_b=kernels[:,offset*(i+1)-3:offset*(i+1)]
            imgs=torch.cat([F.conv2d(img.unsqueeze(0), weight=w, bias=b, stride=1, padding=1,groups=args.groups) for w,b,img in zip(k_w,k_b,imgs)])
        else:
            raise Exception(f"The learned convolution group number \'groups={args.groups}\' is not supported!") 

    return imgs

        
        
        
        
        
                