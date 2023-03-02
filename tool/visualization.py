import os
import random
import time
import cv2
import numpy as np
import logging
import argparse

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn import datasets
# from torch.utils.tensorboard import SummaryWriter

from model import *
from util import dataset
from util import transform, config
from util.util import AverageMeter, intersectionAndUnionGPU

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--config', type=str, default='data/config/pascal/pascal_drnetv3_linit2_aspp_split0_resnet50.yaml', help='config file')
    parser.add_argument('opts', help='see config/ade20k/ade20k_pspnet50.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)


def main_process():
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def main():
    args = get_parser()
    assert args.classes > 1
    assert args.zoom_factor in [1, 2, 4, 8]
    assert (args.train_h - 1) % 8 == 0 and (args.train_w - 1) % 8 == 0
    # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    if args.manual_seed is not None:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        random.seed(args.manual_seed)
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False
    if args.multiprocessing_distributed:
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    global args
    args = argss

    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)

    model = eval(args.arch).Model(args)

    global logger, writer
    logger = get_logger()
    # writer = SummaryWriter(args.save_path)
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))
    logger.info(model)
    print(args)

    model = torch.nn.DataParallel(model.cuda())

    if args.weight:
        if os.path.isfile(args.weight):
            logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            logger.info("=> no weight found at '{}'".format(args.weight))
            raise Exception("'no weight found at '{}'".format(args.weight))

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    assert args.split in [0, 1, 2, 3, 999]

    if args.resized_val:
        val_transform = transform.Compose([
            transform.Resize(size=args.val_size),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)])
    else:
        val_transform = transform.Compose([
            transform.test_Resize(size=args.val_size),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)])
    val_data = dataset.SemData(split=args.split, shot=args.shot, data_root=args.data_root,
                               data_list=args.val_list, transform=val_transform, mode='val',
                               use_coco=args.use_coco, use_split_coco=args.use_split_coco)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False,
                                             num_workers=args.workers, pin_memory=True, sampler=None)
    validate(val_loader, model, criterion, args)


def validate(val_loader, model, criterion, args):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    model_time = AverageMeter()
    data_time = AverageMeter()
    if args.use_coco:
        split_gap = 20
    else:
        split_gap = 5

    if args.manual_seed is not None and args.fix_random_seed_val:
        torch.cuda.manual_seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        random.seed(args.manual_seed)

    model.eval()
    end = time.time()
    if args.split != 999:
        if args.use_coco:
            test_num = 20000
        else:
            test_num = 1000
    else:
        test_num = len(val_loader)
    assert test_num % args.batch_size_val == 0
    iter_num = 0
    total_time = 0
    for i, (input, target, s_input, s_mask, subcls, ori_label) in enumerate(val_loader):
        if (iter_num - 1) * args.batch_size_val >= test_num:
            break
        iter_num += 1
        data_time.update(time.time() - end)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        ori_label = ori_label.cuda(non_blocking=True)
        start_time = time.time()
        output = model(s_x=s_input, s_y=s_mask, x=input, y=target)
        total_time = total_time + 1
        model_time.update(time.time() - start_time)

        visualization(s_x=s_input, s_y=s_mask, x=input, y=target, pred=output, iter=i)



def get_ori_img(img):
    mean = [0.485, 0.456, 0.406]
    mean = [item * 255 for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * 255 for item in std]

    return img * std + mean


def to_numpy(img, label):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))

    if isinstance(label, torch.Tensor):
        label = label.detach().cpu().numpy()

    return img, label


def fuse_img_label(img, label):
    label = label.copy()
    label[label == 255] = 0
    label = np.expand_dims(label, -1)
    label = label.repeat(3, -1) * [255, 0, 0]

    if img.shape != label.shape:
        img = cv2.resize(img, label.shape[:2])

    out = cv2.addWeighted(img.astype('uint8'), 1, label.astype('uint8'), 0.8, 0)
    # out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    return out


def get_heatmap(img, target):
    img = img.detach().cpu().numpy().squeeze()
    target = target.squeeze()

    img = (img - img.min()) / (img.max() - img.min()) * 255
    img[target == 255] = 0
    img = img.astype('uint8')
    heatmap = cv2.applyColorMap(img, cv2.COLORMAP_JET)

    return heatmap


def visualization(s_x, s_y, x, y, pred, iter):
    _, color = datasets.make_s_curve(1000, random_state=0)

    save_path = args.save_path
    save_path = save_path.replace('model', 'visual')

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    s_x, s_y = to_numpy(s_x[0][0], s_y[0][0])
    x, y = to_numpy(x[0], y)
    pred = pred.detach().cpu().numpy()

    s_x = get_ori_img(s_x)
    x = get_ori_img(x)

    sup = fuse_img_label(s_x, s_y)
    plt.subplot(121)
    plt.imshow(sup)
    plt.xticks([])
    plt.yticks([])
    # que = fuse_img_label(x, y)
    # pre = fuse_img_label(x, pred)
    tsne = TSNE()
    pred_tsne = tsne.fit_transform(pred)
    pred_min, pred_max = pred_tsne.min(0), pred_tsne.max(0)
    pred_norm = (pred_tsne - pred_min) / (pred_max - pred_min)

    plt.subplot(122)
    plt.scatter(pred_norm[:, 0], pred_norm[:, 1])
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f'{save_path}/{iter}.png')
    plt.close()
    logger.info(f'Save figure {iter}')
    # cv2.imwrite(f'{save_path}/{iter}_sup.png', sup)
    # cv2.imwrite(f'{save_path}/{iter}_que.png', que)
    # cv2.imwrite(f'{save_path}/{iter}_pre.png', pre)



if __name__ == '__main__':
    main()
