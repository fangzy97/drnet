import argparse
import json
import os
import shutil
import sys
import pickle

import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

import my_optim
from datasets.coco.customized import coco_fewshot
from datasets.coco.transforms import ToTensorNormalize, Resize, RandomMirror
from utils import AverageMeter
from utils.Restore import restore
from utils.para_number import get_model_para_number
from utils.utils import CLASS_LABELS
from oneshot import *

ROOT_DIR = '/'.join(os.getcwd().split('/'))
print(ROOT_DIR)

SNAPSHOT_DIR = os.path.join(ROOT_DIR, 'snapshots', 'coco')
IMG_DIR = os.path.join('/dev/shm/', 'IMAGENET_VOC_3W/imagenet_simple')

LR = 1e-5
BATCH_SIZE = 1


def get_arguments():
    parser = argparse.ArgumentParser(description='OneShot')
    parser.add_argument("--arch", type=str, default='drnet')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=30001)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--dataset", type=str, default="VOC")
    parser.add_argument("--disp_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=30000)
    parser.add_argument("--snapshot_dir", type=str, default=SNAPSHOT_DIR)
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--start_count", type=int, default=0)
    parser.add_argument("--center", type=bool, default=False)
    parser.add_argument("--split", type=str, default='mlclass_train')
    parser.add_argument("--group", type=int, default=0)
    parser.add_argument('--num_folds', type=int, default=4)
    parser.add_argument("--weight-decay", type=float, default=0.0005,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="Momentum component of the optimiser.")
    return parser.parse_args()


def save_checkpoint(args, state, is_best, filename='checkpoint.pth.tar'):
    savedir = os.path.join(
        args.snapshot_dir,
        args.arch + '_batch_' + str(args.batch_size),
        'group_%d_of_%d' % (args.group, args.num_folds)
    )
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    savepath = os.path.join(savedir, filename)
    torch.save(state, savepath)
    if is_best:
        shutil.copyfile(savepath, os.path.join(args.snapshot_dir, 'model_best.pth.tar'))


def get_model(args):
    model = eval(args.arch).OneModel(args)
    model = model.cuda()
    print('Number of Parameters: %d' % (get_model_para_number(model)))

    opti_A = my_optim.get_dconv_finetune_optimizer(args, model)

    snapshot_dir = os.path.join(args.snapshot_dir, 'dconv1', 'group_%d_of_%d' % (args.group, args.num_folds))
    print(args.resume)
    if args.resume:
        restore(snapshot_dir, model)
        print("Resume training...")

    return model, opti_A


def get_save_dir(args):
    snapshot_dir = os.path.join(
        args.snapshot_dir,
        args.arch + '_batch_' + str(args.batch_size),
        'group_%d_of_%d' % (args.group, args.num_folds))
    return snapshot_dir


def train(args):
    hc_loss = 0
    losses = AverageMeter()
    fblosses = AverageMeter()
    olosses = AverageMeter()
    model, optimizer = get_model(args)
    # schedule = StepLR(optimizer, 10000, 0.1, -1)
    # schedule = MultiStepLR(optimizer, [5000, 15000, 25000], gamma=0.1)
    # model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    model.train()

    if not os.path.exists(args.snapshot_dir):
        os.mkdir(args.snapshot_dir)

    cache_dir = 'cache'
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    coco_cache = os.path.join(cache_dir, 'coco_group{}_1shot.pkl'.format(args.group))
    if os.path.exists(coco_cache):
        with open(coco_cache, 'rb') as f:
            coco = pickle.load(f)
    else:
        labels = CLASS_LABELS['COCO'][args.group]
        transforms = Compose([Resize(size=(321, 321))])
        coco = coco_fewshot(
            base_dir='/home/zhiyuan/dataset/coco',
            split='train',
            transforms=transforms,
            to_tensor=ToTensorNormalize(),
            labels=labels,
            n_ways=1,
            n_shots=1,
            max_iters=args.max_steps * args.batch_size,
            n_queries=1
        )
        with open(coco_cache, 'wb') as f:
            pickle.dump(coco, f)

    train_loader = DataLoader(
        coco,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1
    )

    if not os.path.exists(get_save_dir(args)):
        os.makedirs(get_save_dir(args))

    save_log_dir = get_save_dir(args)
    log_file = open(os.path.join(save_log_dir, 'log.txt'), 'w')

    count = args.start_count
    for dat in train_loader:
        if count > args.max_steps:
            break

        sup_img = torch.cat([torch.cat([shot.cuda() for shot in way]) for way in dat['support_images']])
        sup_fg_mask = torch.cat([torch.cat([shot['fg_mask'].float().cuda() for shot in way]) for way in dat['support_mask']])
        sup_fg_mask = sup_fg_mask.unsqueeze(1)

        que_img = torch.cat([que.cuda() for que in dat['query_images']])
        que_labs = torch.cat([que_lab.long().cuda() for que_lab in dat['query_labels']], dim=0)
        que_labs = que_labs.unsqueeze(1)

        optimizer.zero_grad()
        logits = model(que_img, sup_img, sup_img, sup_fg_mask)
        p, mask, loss_val, cluster_loss, loss_bce = model.get_loss(logits, que_labs)

        loss_val_float = loss_val.data.item()
        fb_loss_val = cluster_loss.data.item()
        o_loss_val = loss_bce.data.item()
        hc_loss += loss_bce.data.item()
        losses.update(loss_val_float, 1)
        fblosses.update(fb_loss_val, 1)
        olosses.update(o_loss_val, 1)

        out_str = '%d, %.4f\n' % (count, loss_val_float)
        log_file.write(out_str)
        loss_val.backward()
        optimizer.step()
        # schedule.step()
        # print(losses.avg)
        count += 1
        # print('Step:%d\tLoss:%.3f' % (count, losses.avg))
        if count % args.disp_interval == 0:
            print('Step:%d \t Loss:%.3f \t '
                  'Part1: %.3f \t Part2: %.3f' % (count, losses.avg,
                                                  fblosses.avg, hc_loss / 100))
            hc_loss = 0

        if (count % args.save_interval == 0 or count == 30000 or count == 20000) and count > 0:
            save_checkpoint(args,
                            {
                                'global_counter': count,
                                'state_dict': model.state_dict(),
                                'optimizer': optimizer.state_dict()
                            }, is_best=False,
                            filename='step_%d.pth.tar'
                                     % (count))
    log_file.close()


if __name__ == '__main__':
    args = get_arguments()
    print('Running parameters:\n')
    print(json.dumps(vars(args), indent=4, separators=(',', ':')))
    if not os.path.exists(args.snapshot_dir):
        os.mkdir(args.snapshot_dir)
    train(args)
