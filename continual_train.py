import argparse
import os.path as osp
import sys
import wandb

from torch.backends import cudnn
import torch.nn as nn
import random
from config import cfg
# from reid.evaluators import Evaluator
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from reid.utils.lr_scheduler import WarmupMultiStepLR
from reid.utils.feature_tools import *
from reid.models.layers import DataParallel
from reid.models.resnet import make_model, JointModel
from reid.trainer import Trainer
from torch.utils.tensorboard import SummaryWriter

from lreid_dataset.datasets.get_data_loaders import build_data_loaders
from tools.Logger_results import Logger_res
from reid.evaluation.fast_test import fast_test_p_s
from reid.models.rehearser import KernelLearning
from reid.models.cm import ClusterMemory
import datetime
from torch.autograd import Variable
import numpy as np
import copy
def cur_timestamp_str():
    now = datetime.datetime.now()
    year = str(now.year)
    month = str(now.month).zfill(2)
    day = str(now.day).zfill(2)
    hour = str(now.hour).zfill(2)
    minute = str(now.minute).zfill(2)

    content = "{}-{}{}-{}{}".format(year, month, day, hour, minute)
    return content
def main():
    args = parser.parse_args()

    if args.seed is not None:
        print("setting the seed to",args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    cfg.merge_from_file(args.config_file)
    main_worker(args, cfg)


def main_worker(args, cfg):
    # log_name = 'log.txt'
    timestamp = cur_timestamp_str()
    wandb.init(project="AAAI2025-LReID-DASK", name=f"setting_{args.setting}_{timestamp}", config=args)
    log_name = f'log_{timestamp}.txt'
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, log_name))
    else:
        log_dir = osp.dirname(args.test_folder)
        sys.stdout = Logger(osp.join(log_dir, log_name))
    print("==========\nArgs:{}\n==========".format(args))
    log_res_name=f'log_res_{timestamp}.txt'
    logger_res=Logger_res(osp.join(args.logs_dir, log_res_name))    # record the test results
    

    """
    loading the datasets:
    setting： 1 or 2 
    """
    if 1 == args.setting:
        training_set = ['market1501', 'cuhk_sysu','dukemtmc', 'msmt17','cuhk03'] #,'sense','cuhk_sysu',

    #数据集大小
    elif 2 == args.setting:
        training_set = ['cuhk03', 'msmt17', 'cuhk_sysu', 'market1501','dukemtmc']
    #先MSMT17最复杂的场景 再按数据集大小
    elif 3 == args.setting:
        training_set = ['msmt17', 'cuhk03', 'cuhk_sysu', 'market1501','dukemtmc']
    elif 4 == args.setting:
        training_set = ['dukemtmc', 'market1501', 'cuhk03', 'msmt17', 'cuhk_sysu']
    elif 5 == args.setting:
        training_set = ['cuhk_sysu', 'dukemtmc', 'cuhk03', 'msmt17', 'market1501']
    elif 6 == args.setting:
        training_set = ['cuhk03', 'msmt17', 'cuhk_sysu', 'market1501','dukemtmc' ]
    elif 7 == args.setting:
        training_set = ['market1501', 'msmt17', 'dukemtmc', 'cuhk_sysu', 'cuhk03']
    # all the revelent datasets
    all_set = ['market1501', 'dukemtmc', 'msmt17',
               'sense', 'grid', 'cuhk03','prid']  # 'sense','prid', 'cuhk_sysu','cuhk01', 'cuhk02', 'grid', 'viper', 'ilids','cuhk_sysu'
    # the datsets only used for testing
    testing_only_set = [x for x in all_set if x not in training_set]
    # get the loders of different datasets
    all_train_sets, all_test_only_sets = build_data_loaders(args, training_set, testing_only_set)    
    
    first_train_set = all_train_sets[0]
    model = make_model(args, num_class=first_train_set[1], camera_num=0, view_num=0)

    model.cuda()
    model = DataParallel(model)    
    writer = SummaryWriter(log_dir=args.logs_dir)
    # Load from checkpoint
    '''test the models under a folder'''
    if args.test_folder:
        ckpt_name = [x + '_checkpoint.pth.tar' for x in training_set]   # obatin pretrained model name
        checkpoint = load_checkpoint(osp.join(args.test_folder, ckpt_name[0]))  # load the first model
        copy_state_dict(checkpoint['state_dict'], model)     #   
        rehearser_list=[] 
        for step in range(len(ckpt_name) - 1):
            model_old = copy.deepcopy(model)    # backup the old model            
            checkpoint = load_checkpoint(osp.join(args.test_folder, ckpt_name[step + 1]))
            copy_state_dict(checkpoint['state_dict'], model)

                
            if args.fix_EMA>=0:
                best_alpha=args.fix_EMA
            else:
                best_alpha = get_adaptive_alpha(args, model, model_old, all_train_sets, step + 1)

            model = linear_combination(args, model, model_old, best_alpha)

            save_name = '{}_checkpoint_adaptive_ema_{:.4f}.pth.tar'.format(training_set[step+1], best_alpha)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': 0,
                'mAP': 0,
            }, True, fpath=osp.join(args.logs_dir, save_name))            
            
            # load the learned AKPNet models
            if args.mobile:
                rehearser=KernelLearning(n_kernel=args.n_kernel, groups=args.groups, model='mobile-v3').cuda()
                checkpoint = load_checkpoint('rehearser_pretrain_learn_kernel_c{}-g{}_mobilenet-v3/{}_rehearser_49.pth.tar'.format(args.n_kernel,args.groups,
                                                                                                                                    all_train_sets[step+1][-1])) 
            else:
                rehearser=KernelLearning(n_kernel=args.n_kernel, groups=args.groups, model='shufflenet_v2').cuda()
                checkpoint = load_checkpoint('rehearser_pretrain_learn_kernel_c{}-g{}/{}_rehearser_49.pth.tar'.format(args.n_kernel,args.groups,
                                                                                                                all_train_sets[step+1][-1])) 
                        
            copy_state_dict(checkpoint['state_dict'], rehearser)     #   
            rehearser_list.append(rehearser)
        if args.joint_test:
            test_model = JointModel(args=args,model1=rehearser_list[-1], model2=model)
        else:
            test_model=model
        fast_test_p_s(test_model, all_train_sets, all_test_only_sets, set_index=len(all_train_sets)-1, logger=logger_res,
                      args=args,writer=writer)

        exit(0)
    

    # resume from a model
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        copy_state_dict(checkpoint['state_dict'], model)
        start_epoch = checkpoint['epoch']
        best_mAP = checkpoint['mAP']
        print("=> Start epoch {}  best mAP {:.1%}".format(start_epoch, best_mAP))
   
    # Evaluator
    if args.MODEL in ['50x']:
        out_channel = 2048
    else:
        raise AssertionError(f"the model {args.MODEL} is not supported!")


    # train on the datasets squentially
    rehearser_list=[]
    fisher_accum = None  # Store accumulated Fisher Information
    
    for set_index in range(0, len(training_set)):       
        model_old = copy.deepcopy(model)

        if args.resume != '' and set_index==0:
            continue
        model, current_fisher = train_dataset(cfg, args, all_train_sets, all_test_only_sets, set_index, model, out_channel,
                                            writer,logger_res=logger_res,rehearser_list=rehearser_list, prev_fisher=fisher_accum)
        
        # Accumulate Fisher Information
        if args.fisher_freeze:
            if fisher_accum is None:
                fisher_accum = current_fisher
            else:
                for n, p in fisher_accum.items():
                    if n in current_fisher:
                        # Handle potential shape mismatch for classifier if we were tracking it (though usually we exclude head)
                        # Here we simply add if shapes match. If param shape changed (e.g. classifier), 
                        # we technically start fresh for the new parts or handle it.
                        # For simplicity, if shapes match, add. If not, replace or ignore?
                        # EWC usually cares about 'shared' body.
                        if p.shape == current_fisher[n].shape:
                            fisher_accum[n] += current_fisher[n]
                        else:
                            # If shape changed (classifier), arguably old fisher is invalid or we pad it.
                            # For freezing based on importance, we usually care about the backbone.
                            # We'll just replace disparate shapes with new (or handle partial sum if we implemented slicing).
                             # For now, let's just keep the matching parts or overwrite.
                            # For classifier growth, usually we rely on the fact that old weights are copied to the top.
                            # We will just overwrite for simplicity in this implementation, 
                            # assuming Backbone params don't change shape.
                            if p.shape[0] < current_fisher[n].shape[0]:
                                # If expanded, we can pad the old fisher? 
                                # But straightforward accumulation is tricky with shape change.
                                # Let's just track backbone parameters for freezing to avoid complexity.
                                pass
                            fisher_accum[n] = current_fisher[n]
                    else:
                        fisher_accum[n] = current_fisher[n]
                # Add new params
                for n, p in current_fisher.items():
                    if n not in fisher_accum:
                        fisher_accum[n] = p

        if set_index>0:
            # best_alpha = get_adaptive_alpha(args, model, model_old, all_train_sets, set_index)
            # if args.fix_EMA>=0:
            #     best_alpha=args.fix_EMA
            # model = linear_combination(args, model, model_old, best_alpha)   
            fast_test_p_s(model, all_train_sets, all_test_only_sets, set_index=set_index, logger=logger_res,
                      args=args,writer=writer)
    print('finished')
def get_normal_affinity(x,Norm=100):
    from reid.metric_learning.distance import cosine_similarity
    pre_matrix_origin=cosine_similarity(x,x)
    pre_matrix_origin=-100*torch.eye(x.size(0)).to(x.device)+pre_matrix_origin
    pre_affinity_matrix=F.softmax(pre_matrix_origin*Norm, dim=1)
    return pre_affinity_matrix
def get_adaptive_alpha(args, model, model_old, all_train_sets, set_index):
    dataset_new, num_classes_new, train_loader_new, _, init_loader_new, name_new = all_train_sets[
        set_index]  # trainloader of current dataset
    features_all_new, labels_all, fnames_all, camids_all, features_mean_new, labels_named = extract_features_voro(model,
                                                                                                          init_loader_new,
                                                                                                          get_mean_feature=True)
    features_all_old, _, _, _, features_mean_old, _ = extract_features_voro(model_old,init_loader_new,get_mean_feature=True)

    features_all_new=torch.stack(features_all_new, dim=0)
    features_all_old=torch.stack(features_all_old,dim=0)
    Affin_new = get_normal_affinity(features_all_new, args.global_alpha)
    Affin_old = get_normal_affinity(features_all_old, args.global_alpha)

    Difference= torch.abs(Affin_new-Affin_old).sum(-1).mean()
    # 相似度作为融合参数
    sim=(Affin_new*Affin_old).sum(-1).mean()
    alpha=sim
    if args.absolute_delta:
        alpha=float(1-Difference)
    
    return alpha

def obtain_old_types(args, all_train_sets, set_index, model):
    dataset_old, num_classes_old, train_loader_old, _, init_loader_old, name_old = all_train_sets[
        set_index]  # trainloader of old dataset
    features_all_old, labels_all_old, fnames_all, camids_all, features_mean, labels_named, vars_mean = extract_features_proto(model,
                                                                                                                  init_loader_old,
                                                                                                                  get_mean_feature=True)  # init_loader is original designed for classifer init
    features_all_old = torch.stack(features_all_old)
    labels_all_old = torch.tensor(labels_all_old).to(features_all_old.device)
    features_all_old.requires_grad = False
    # savename=osp.join(args.logs_dir, f"proto_{name_old}.pt")
    # torch.save(,savename)
    return features_all_old, labels_all_old, features_mean, labels_named,vars_mean

def compute_fisher_matrix(model, data_loader, num_samples=1000):
    print("Computing Fisher Information Matrix...")
    fisher = {}
    model.eval()
    
    # Ensure gradients are enabled for Fisher computation
    for param in model.parameters():
        param.requires_grad = True
        
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0) # Dummy optimizer
    
    # We need a loss function. We'll use the model's output prediction confidence (standard EWC approximation)
    # or just use the ground truth loss if available.
    # Using ground truth labels on training data is better for 'Empirical Fisher'.
    # Note: data_loader iteration
    
    count = 0
    # Collect named parameters to initialize fisher dict keys
    params_dict = {n: p for n, p in model.named_parameters() if p.requires_grad}
    for n, p in params_dict.items():
        fisher[n] = torch.zeros_like(p.data)
        
    for i, inputs in enumerate(data_loader):
        # We need to unpack inputs based on existing code structure
        # _parse_data is in Trainer, but here we can just do minimal unpacking
        # inputs format: imgs, imgs_origin, _, pids, cids, domains
        if len(inputs) == 6:
            imgs, _, _, pids, _, _ = inputs
        else:
            # Fallback or different loader type
            imgs = inputs[0]
            pids = inputs[3]
            
        imgs = imgs.cuda()
        pids = pids.cuda()
        
        batch_size = imgs.size(0)
        
        # Forward
        model.zero_grad()
        # model forward signature: x, domains=None, training_phase=None, get_all_feat=False, epoch=0, head_only=False
        # output: global_feat, bn_feat, cls_outputs, x (if get_all_feat=True/False checks)
        # We need cls_outputs for loss
        # Note: model.module is likely since DataParallel is used.
        # But we passed `model` which might be DataParallel.
        
        # We need to call forward such that it returns logits.
        # Looking at ResNet.forward:
        # if self.training is False: return global_feat
        # if self.training is True: return global_feat, bn_feat, cls_outputs, x
        
        # So we must set model.train() to get logits, but we don't want to update BN stats.
        model.train() 
        # Freeze BN stats
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                
        # Forward
        outputs = model(imgs)
        # Outputs is (global_feat, bn_feat, cls_outputs, x)
        if isinstance(outputs, tuple):
            cls_outputs = outputs[2]
        else:
            # Should not happen based on code analysis if training=True
            cls_outputs = outputs
            
        # Standard Empirical Fisher: Gradient of log-likelihood of GROUND TRUTH
        # L = CE(y_pred, y_true). Gradient is dL/dtheta. Fisher ~ E[(dL/dtheta)^2]
        loss = F.cross_entropy(cls_outputs, pids)
        loss.backward()
        
        for n, p in model.named_parameters():
             if p.grad is not None:
                 fisher[n] += p.grad.data.pow(2) * batch_size
                 
        count += batch_size
        if count >= num_samples:
            break
            
    # Normalize
    for n in fisher:
        fisher[n] /= count
        
    print(f"Fisher Matrix Computed on {count} samples.")
    return fisher

def train_dataset(cfg, args, all_train_sets, all_test_only_sets, set_index, model, out_channel, writer,logger_res=None,rehearser_list=None, prev_fisher=None):
    if  set_index>0:              
        # load the learned AKPNet models
        if args.mobile:
            rehearser=KernelLearning(n_kernel=args.n_kernel, groups=args.groups, model='mobile-v3').cuda()
        else:
            rehearser=KernelLearning(n_kernel=args.n_kernel, groups=args.groups, model='shufflenet_v2').cuda()
        
        if args.mobile:                    
            checkpoint = load_checkpoint('rehearser_pretrain_learn_kernel_c{}-g{}_mobilenet-v3/{}_rehearser_49.pth.tar'.format(args.n_kernel,args.groups,
                                                                                                                                all_train_sets[set_index-1][-1]))  #
        else:
            checkpoint = load_checkpoint('rehearser_pretrain_learn_kernel_c{}-g{}/{}_rehearser_49.pth.tar'.format(args.n_kernel,args.groups,
                                                                                                                all_train_sets[set_index-1][-1]))  # load the first model
        
        copy_state_dict(checkpoint['state_dict'], rehearser)     #   
        rehearser_list.append(rehearser)    
    else:
        rehearser=None
    dataset, num_classes, train_loader, test_loader, init_loader, name = all_train_sets[
        set_index]  # status of current dataset    

    Epochs= args.epochs0 if 0==set_index else args.epochs          

    if set_index<=1:
        add_num = 0
        old_model=None
    else:
        add_num = sum(
            [all_train_sets[i][1] for i in range(set_index - 1)])  # get person number in existing domains
    
    
    if set_index>0:
        '''store the old model'''
        old_model = copy.deepcopy(model)
        old_model = old_model.cuda()
        old_model.eval()

        # after sampling rehearsal, recalculate the addnum(historical ID number)
        add_num = sum([all_train_sets[i][1] for i in range(set_index)])  # get model out_dim
        # Expand the dimension of classifier
        org_classifier_params = model.module.classifier.weight.data
        model.module.classifier = nn.Linear(out_channel, add_num + num_classes, bias=False)
        model.module.classifier.weight.data[:add_num].copy_(org_classifier_params)
        model.cuda()    
        # Initialize classifer with class centers    
        class_centers = initial_classifier(model, init_loader)
        model.module.classifier.weight.data[add_num:].copy_(class_centers)
        model.cuda()

    # Re-initialize optimizer
    params = []
    for key, value in model.named_params(model):
        if not value.requires_grad:
            print('not requires_grad:', key)
            continue
        params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(params, momentum=args.momentum)    
    Stones=args.milestones
    lr_scheduler = WarmupMultiStepLR(optimizer, Stones, gamma=0.1, warmup_factor=0.01, warmup_iters=args.warmup_step)
    
    
    
    # Apply Fisher-based Freezing Mask
    grad_mask = None
    if args.fisher_freeze and prev_fisher is not None:
        print("Applying Fisher-based Freezing (EWC-style)...")
        hooks = [] # We don't use hooks anymore but keeping var to avoid modifying too much logic structure if confused, but actually we remove hook logic.
        
        all_fisher_vals = []
        for n, p in model.named_parameters():
            if n in prev_fisher:
                if prev_fisher[n].shape == p.shape:
                    all_fisher_vals.append(prev_fisher[n].flatten())
                elif 'classifier' in n:
                     min_len = min(prev_fisher[n].shape[0], p.shape[0])
                     all_fisher_vals.append(prev_fisher[n][:min_len].flatten())
        
        if len(all_fisher_vals) > 0:
            flat_fisher = torch.cat(all_fisher_vals)
            k = int(len(flat_fisher) * args.fisher_ratio)
            if k > 0:
                threshold = torch.topk(flat_fisher, k, largest=True)[0][-1]
                print(f"Fisher Threshold: {threshold.item()} (Top {args.fisher_ratio*100}%)")
                
                grad_mask = {}
                count_frozen = 0
                for n, p in model.named_parameters():
                    if n in prev_fisher:
                        f_val = prev_fisher[n]
                        if f_val.device != p.device:
                            f_val = f_val.to(p.device)
                            
                        if f_val.shape != p.shape:
                            # Handling shape mismatch (classifier expansion)
                            full_mask = torch.ones_like(p, dtype=torch.float32)
                            if p.ndim == 2:
                                min_rows = min(f_val.shape[0], p.shape[0])
                                min_cols = min(f_val.shape[1], p.shape[1])
                                sub_mask = (f_val[:min_rows, :min_cols] < threshold).float()
                                full_mask[:min_rows, :min_cols] = sub_mask
                            elif p.ndim == 1:
                                min_len = min(f_val.shape[0], p.shape[0])
                                sub_mask = (f_val[:min_len] < threshold).float()
                                full_mask[:min_len] = sub_mask
                            
                            grad_mask[n] = full_mask
                        else:
                            grad_mask[n] = (f_val < threshold).float()
                        
                        # Count frozen elements
                        count_frozen += (grad_mask[n] == 0).sum().item()
                            
                print(f"Gradient mask prepared. Approx {count_frozen} parameters frozen.")
            else:
                print("Fisher Ratio results in 0 parameters to freeze.")
        else:
            print("No matching Fisher params found for freezing.")

    trainer = Trainer(cfg, args, model, add_num + num_classes,  writer=writer, grad_mask=grad_mask)

    

    print('####### starting training on {} #######'.format(name))
    
    # Initialize EMA model
    model_ema = copy.deepcopy(model)
    model_ema.eval()

    for epoch in range(0, Epochs):
        if args.random_rehearser and set_index>0:
            rehearser= random.choice(rehearser_list)
        elif set_index>0:
            rehearser=rehearser_list[-1]

        train_loader.new_epoch()
        trainer.train(epoch, train_loader,  optimizer, training_phase=set_index + 1,
                      train_iters=len(train_loader), add_num=add_num, old_model=old_model,rehearser=rehearser
                      )
        lr_scheduler.step()       

        # Dynamic EMA Update
        # Lambda increases with epoch: Early -> small lambda (fast adapt), Late -> large lambda (stability)
        # Range: 0.1 to 0.95
        lambda_ema = 0.1 + (0.95 - 0.1) * (epoch / Epochs)
        
        # Update EMA model
        for param_q, param_k in zip(model.parameters(), model_ema.parameters()):
            param_k.data = param_k.data * lambda_ema + param_q.data * (1. - lambda_ema)
       

        if ((epoch + 1) % args.eval_epoch == 0 or epoch+1==Epochs):
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'mAP': 0.,
            }, True, fpath=osp.join(args.logs_dir, '{}_checkpoint.pth.tar'.format(name)))

            logger_res.append('epoch: {}'.format(epoch + 1))
            
            mAP=0.
            if args.middle_test or epoch+1==Epochs:
                # mAP = test_model(model, all_train_sets, all_test_only_sets, set_index, logger_res=logger_res)    
                mAP = fast_test_p_s(model, all_train_sets, all_test_only_sets, set_index=set_index, logger=logger_res,
                      args=args,writer=writer)                
          
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'mAP': mAP,
            }, True, fpath=osp.join(args.logs_dir, '{}_checkpoint.pth.tar'.format(name)))    
            
    # Compute new Fisher
    curr_fisher = {}
    if args.fisher_freeze:
        curr_fisher = compute_fisher_matrix(model_ema, init_loader, num_samples=args.fisher_sample_num)

    return model_ema, curr_fisher 


def linear_combination(args, model, model_old, alpha, model_old_id=-1):
    print("*******combining the models with alpha: {}*******".format(alpha))
    '''old model '''
    model_old_state_dict = model_old.state_dict()
    '''latest trained model'''
    model_state_dict = model.state_dict()

    ''''create new model'''
    model_new = copy.deepcopy(model)
    model_new_state_dict = model_new.state_dict()
    '''fuse the parameters'''
    for k, v in model_state_dict.items():
        if model_old_state_dict[k].shape == v.shape:
            # print(k,'+++')
                model_new_state_dict[k] = alpha * v + (1 - alpha) * model_old_state_dict[k]
        else:
            print(k, '...')
            num_class_old = model_old_state_dict[k].shape[0]
            model_new_state_dict[k][:num_class_old] = alpha * v[:num_class_old] + (1 - alpha) * model_old_state_dict[k]
    model_new.load_state_dict(model_new_state_dict)
    return model_new


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Continual training for lifelong person re-identification")
    # data
    parser.add_argument('-b', '--batch-size', type=int, default=128)
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # model    
    parser.add_argument('--MODEL', type=str, default='50x',
                        choices=['50x'])
    # optimizer
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam'],
                        help="optimizer ")
    parser.add_argument('--lr', type=float, default=0.008,
                        help="learning rate of new parameters, for pretrained ")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--warmup-step', type=int, default=10)
    parser.add_argument('--milestones', nargs='+', type=int, default=[30],
                        help='milestones for the learning rate decay')
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs0', type=int, default=80)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--eval_epoch', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--print-freq', type=int, default=200)
    
    # path   
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default='/DATA2025/cjh/AAAI2025-LReID-DASK/PRID')
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join('/DATA2025/cjh/AAAI2025-LReID-DASK/output'))

    parser.add_argument('--config_file', type=str, default='config/base.yml',
                        help="config_file")
  
    parser.add_argument('--test_folder', type=str, default=None, help="test the models in a file")
   
    parser.add_argument('--setting', type=int, default=1, choices=[1, 2,3,4,5,6,7], help="training order setting")
    parser.add_argument('--middle_test', action='store_true', help="test during middle step")
    parser.add_argument('--AF_weight', default=1.0, type=float, help="anti-forgetting weight")   

    parser.add_argument('--fix_EMA', default=0.5, type=float, help="model fusion weight") 
   
    parser.add_argument('--global_alpha',  type=float, default=100,  help="")   
    parser.add_argument('--absolute_feat',  action='store_true', help="")        
    parser.add_argument('--save_evaluation', action='store_true', help="save ranking results")
    parser.add_argument('--absolute_delta', action='store_true',default=True, help="only use dual teacher")
    parser.add_argument('--trans', action='store_true',default=True, help="only use dual teacher")

    # parser.add_argument('--color_style', type=str,default='rgb', help="select the color", choices=['lab','rgb'])
    # parser.add_argument('--learn_kernel', action='store_true', help="learnable style transfer kernel")
    parser.add_argument('--random_rehearser', action='store_true', help="select a random rehearser for data augmentation")
    parser.add_argument('--blur', action='store_true', help="adopt blur augmentation")
    parser.add_argument('--n_kernel', default=1, type=int, help="number of Distribution Transfer kernel")   
    parser.add_argument('--groups', default=1, type=int, help="convolution group number of each Distribution Transfer kernel")  
    parser.add_argument('--joint_test', action='store_true', help="use the AKPNet model during testing")   
    parser.add_argument('--mobile', action='store_true', help="use the mobilenet-v3 as the backbone of synthetic models") 
    parser.add_argument('--aux_weight', default=4.5, type=float, help="the loss weight of rehearsed data, e.g. β in the paper")
    parser.add_argument('--dropout', default=0.5, type=float, help="dropout probability")
    parser.add_argument('--l2sp-weight', default=0.0, type=float, help="L2-SP regularization weight")
    parser.add_argument('--re-ranking', action='store_true', help="Enable k-reciprocal re-ranking for testing")
    
    # Fisher Freezing 
    parser.add_argument('--fisher-freeze', action='store_true', help="Enable Fisher-based parameter freezing")
    parser.add_argument('--fisher-ratio', default=0.3, type=float, help="Ratio of parameters to freeze (highest Fisher info)")
    parser.add_argument('--fisher-sample-num', default=1000, type=int, help="Number of samples to use for Fisher computation")
    parser.add_argument('--accumulation-steps', default=1, type=int, help="gradient accumulation steps")
    main()

