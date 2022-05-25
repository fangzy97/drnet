import argparse
import json
import os
import shutil
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import random

import my_optim
from oneshot import *
from utils import AverageMeter
from utils.LoadDataSeg import data_loader
from utils.para_number import get_model_para_number
from utils.Restore import restore


def setup_seed(seed):
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    print("Set the random seed as {}".format(seed))

setup_seed(1234)

# update with your path
# All the jupyter notebooks in the repository already have this
# try:
#     from apex.fp16_utils import *
#     from apex import amp, optimizers
# except ImportError:
#     print('hehe')

ROOT_DIR = '/'.join(os.getcwd().split('/'))
print (ROOT_DIR)


SNAPSHOT_DIR = os.path.join(ROOT_DIR, 'snapshots')
IMG_DIR = os.path.join('/dev/shm/', 'IMAGENET_VOC_3W/imagenet_simple')

LR = 1e-5

def lr_poly(base_lr, iter,max_iter,power=0.9):
    return base_lr*((1-float(iter)/max_iter)**(power))

def get_arguments():
    parser = argparse.ArgumentParser(description='OneShot')
    parser.add_argument("--arch", type=str,default='drnet')
    parser.add_argument("--max_steps", type=int, default=30001)
    parser.add_argument("--lr", type=float, default=LR)
    # parser.add_argument("--decay_points", type=str, default='none')
    parser.add_argument("--disp_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=10000)
    parser.add_argument("--snapshot_dir", type=str, default=SNAPSHOT_DIR)
    # parser.add_argument("--restore_from", type=str, default='')
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--start_count", type=int, default=0)
    parser.add_argument("--center", type=bool, default=False)
    # parser.add_argument("--split", type=str, default='mlclass_train')
    parser.add_argument("--split", type=str, default='mlclass_train')
    parser.add_argument("--group", type=int, default=0)
    parser.add_argument('--num_folds', type=int, default=4)
    parser.add_argument("--weight-decay", type=float, default=0.0005,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="Momentum component of the optimiser.")
    return parser.parse_args()

def save_checkpoint(args, state, is_best, filename='checkpoint.pth.tar'):
    savedir = os.path.join(args.snapshot_dir, args.arch, 'group_%d_of_%d'%(args.group, args.num_folds))
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    savepath = os.path.join(savedir, filename)
    torch.save(state, savepath)
    if is_best:
        shutil.copyfile(savepath, os.path.join(args.snapshot_dir, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, step, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 2 epochs"""
    lr *= (0.1 ** (step // 30000))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_model(args):

    model = eval(args.arch).OneModel(args)

    model = model.cuda()

    print('Number of Parameters: %d'%(get_model_para_number(model)))
    
    opti_A = my_optim.get_dconv_finetune_optimizer(args, model)

    # if os.path.exists(args.restore_from):

    snapshot_dir = os.path.join(args.snapshot_dir, 'dconv1', 'group_%d_of_%d'%(args.group, args.num_folds))
    print(args.resume)
    if args.resume:
        restore(snapshot_dir, model)
        print("Resume training...")

    return model, opti_A

def get_save_dir(args):
    snapshot_dir = os.path.join(args.snapshot_dir, args.arch, 'group_%d_of_%d'%(args.group, args.num_folds))
    return snapshot_dir



def train(args):
    hc_loss = 0
    now_lr = args.lr
    if 'center' in args.arch:
        args.center = True
    losses = AverageMeter()
    fblosses = AverageMeter()
    olosses = AverageMeter()
    model, optimizer= get_model(args)
    # model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    model.train()

    if not os.path.exists(args.snapshot_dir):
        os.mkdir(args.snapshot_dir)

    train_loader = data_loader(args)

    if not os.path.exists(get_save_dir(args)):
        os.makedirs(get_save_dir(args))

    save_log_dir = get_save_dir(args)
    log_file  = open(os.path.join(save_log_dir, 'log.txt'),'w')


    count = args.start_count
    for dat in train_loader:
        if count > args.max_steps:
            break

        if args.split == 'class_train':
            anchor_img, anchor_mask, pos_img, pos_mask, neg_img, neg_mask, category = dat
        else:
            anchor_img, anchor_mask, pos_img, pos_mask, neg_img, neg_mask, category = dat

        anchor_img, anchor_mask, pos_img, pos_mask, neg_mask, \
            = anchor_img.cuda(), anchor_mask.cuda(), pos_img.cuda(), pos_mask.cuda(), neg_mask.cuda()

        if args.group == 0:
            neg_mask = neg_mask - 6
            neg_mask[neg_mask < 0] = 255
            neg_mask[neg_mask>100] = 255
        elif args.group == 1:
            neg_mask[(neg_mask>=6)&(neg_mask<=10)] = 255
            neg_mask[neg_mask<6] -= 1
            neg_mask[neg_mask>10] -= 6
            neg_mask[neg_mask<0] = 255
            neg_mask[neg_mask>100] = 255
        elif args.group == 2:
            neg_mask[(neg_mask>=11)&(neg_mask<=15)] = 255
            neg_mask[neg_mask<11] -= 1
            neg_mask[neg_mask>15] -= 6
            neg_mask[neg_mask<0] = 255
            neg_mask[neg_mask>100] = 255
        else:
            neg_mask[neg_mask>=16] = 255
            neg_mask[neg_mask<16] -= 1
            neg_mask[neg_mask < 0] = 255
            neg_mask[neg_mask > 100] = 255

        anchor_mask = torch.unsqueeze(anchor_mask, dim=1)
        # print(torch.max(anchor_mask))
        pos_mask = torch.unsqueeze(pos_mask, dim=1)
        neg_mask = torch.unsqueeze(neg_mask, dim=1)
        optimizer.zero_grad()
        if args.split == 'class_train':
            logits = model(anchor_img, pos_img, pos_img, pos_mask, category=category, group=args.group)
        else:
            logits = model(anchor_img, pos_img, pos_img, pos_mask, category=category, group=args.group)
        p, mask, loss_val, cluster_loss, loss_bce = model.get_loss(logits, anchor_mask, fb_mask=neg_mask)
        #if count % 10 == 0:
        #    writer = SummaryWriter(log_dir=os.path.join(save_log_dir, 'log'), comment='anchor_img')
        #    ai = anchor_img
        #    am = anchor_mask
        #    nm = neg_mask
        #    ai = vutils.make_grid(ai, normalize=True, scale_each=False)
        #    am = vutils.make_grid(am, normalize=True, scale_each=False)
        #    nm = vutils.make_grid(nm, normalize=True, scale_each=False)
        #    m = vutils.make_grid(mask, normalize=False, scale_each=False)
        #    p = vutils.make_grid(p, normalize=False, scale_each=False)
        #    writer.add_image("anchor", ai, count)
        #    writer.add_image("mask", am, count)
        #    writer.add_image("amask", nm, count)
        #    writer.add_image("aoutput", m, count)
        #    writer.add_image("output", p, count)
        #    writer.close()

        loss_val_float = loss_val.data.item()
        fb_loss_val = cluster_loss.data.item()
        o_loss_val = loss_bce.data.item()
        hc_loss += loss_bce.data.item()
        losses.update(loss_val_float, 1)
        fblosses.update(fb_loss_val, 1)
        olosses.update(o_loss_val, 1)

        out_str = '%d, %.4f\n'%(count, loss_val_float)
        log_file.write(out_str)

        # adjust_learning_rate(optimizer, count, now_lr)
        # now_lr = optimizer.param_groups[0]['lr']

        # lr = lr_poly(base_lr=args.lr, iter=count, max_iter=args.max_steps)
        # optimizer.param_groups[0]['lr'] = lr * 0.1
        # optimizer.param_groups[1]['lr'] = lr * 0.2
        # optimizer.param_groups[2]['lr'] = 0
        # optimizer.param_groups[3]['lr'] = 0
        # optimizer.param_groups[4]['lr'] = lr * 10
        # optimizer.param_groups[5]['lr'] = lr * 20
        # optimizer.param_groups[0]['lr'] = lr
        # optimizer.param_groups[1]['lr'] = lr * 2
        # optimizer.param_groups[2]['lr'] = lr * 10
        # optimizer.param_groups[3]['lr'] = lr * 20

        # my_optim.adjust_learning_rate(optimizer, count)
        # with amp.scale_loss(loss_val, optimizer) as scaled_loss:
        #     scaled_loss.backward()

        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        #                                               lambda step: (1.0 - step / args.total_iters) if step <= args.total_iters else 0,
        #                                               last_epoch=-1)
        #
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30000, gamma=0.1)
        loss_val.backward()
        # if args.center:
        #     for param in model.centerloss.parameters():
        #         # lr_cent is learning rate for center loss, e.g. lr_cent = 0.5
        #         param.grad.data *= (0.5 / (0.002 * args.lr * 10))
        optimizer.step()
        # scheduler.step()


        count += 1
        if count%args.disp_interval == 0:
            # print('Step:%d \t Loss:%.3f '%(count, losses.avg))
            print('Step:%d \t Loss:%.3f \t '
                  'Part1: %.3f \t Part2: %.3f'%(count, losses.avg,
                                        fblosses.avg, hc_loss/100))
            # print(optimizer.state_dict()['param_groups'][0]['lr'])
            hc_loss = 0


        if (count % args.save_interval == 0) and count > 0:
            save_checkpoint(args,
                            {
                                'global_counter': count,
                                'state_dict':model.state_dict(),
                                'optimizer':optimizer.state_dict()
                            }, is_best=False,
                            filename='step_%d.pth.tar'
                                     %(count))
    log_file.close()

if __name__ == '__main__':
    args = get_arguments()
    print ('Running parameters:\n')
    print (json.dumps(vars(args), indent=4, separators=(',', ':')))
    if not os.path.exists(args.snapshot_dir):
        os.mkdir(args.snapshot_dir)
    train(args)
