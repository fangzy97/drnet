import argparse
import logging
import os
import random
import time

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
from model import *
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from util import config, dataset, transform
from util.util import (AverageMeter, fix_bn, intersectionAndUnionGPU,
                       poly_learning_rate)

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--config', type=str, default='data/config/pascal/pascal_drnet_split0_resnet50.yaml', help='config file')
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
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    global args, logger, writer
    args = argss
    args.gpu = gpu

    logger = get_logger()
    writer = SummaryWriter(args.save_path)

    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    
    logger.info("=> creating model ...")
    model = eval(args.arch).Model(args)
    optimizer = model._optimizer(args)

    logger.info("Classes: {}".format(args.classes))
    logger.info(model)
    logger.info(args)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        model = model.cuda()

    if args.weight:
        if os.path.isfile(args.weight):
            logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            logger.info("=> no weight found at '{}'".format(args.weight))

    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))


    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    assert args.split in [0, 1, 2, 3, 999]
    train_transform = [
        transform.RandScale([args.scale_min, args.scale_max]),
        transform.RandRotate([args.rotate_min, args.rotate_max], padding=mean, ignore_label=args.padding_label),
        transform.RandomGaussianBlur(),
        transform.RandomHorizontalFlip(),
        transform.Crop([args.train_h, args.train_w], crop_type='rand', padding=mean, ignore_label=args.padding_label),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)]
    train_transform = transform.Compose(train_transform)
    train_data = dataset.BaseData(split=args.split, data_root=args.data_root,
                                  data_list=args.train_list, data_set=args.data_set, transform=train_transform, mode='train',
                                  use_split_coco=args.use_split_coco, batch_size=args.batch_size)

    train_sampler = train_sampler = DistributedSampler(train_data) if args.distributed else None
    train_loader = torch.utils.data.DataLoader(train_data, 
                                               batch_size=args.batch_size, 
                                               shuffle=(train_sampler is None), 
                                               num_workers=args.workers, 
                                               pin_memory=True, 
                                               sampler=train_sampler, 
                                               drop_last=True)
    if args.evaluate:
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
        val_data = dataset.BaseData(split=args.split, data_root=args.data_root, data_set=args.data_set,
                                   data_list=args.val_list, transform=val_transform, mode='val', 
                                   batch_size=args.batch_size_val, use_split_coco=args.use_split_coco)
        val_sampler = None
        val_loader = torch.utils.data.DataLoader(val_data, 
                                                 batch_size=args.batch_size_val, 
                                                 shuffle=False, 
                                                 num_workers=args.workers, 
                                                 pin_memory=True, 
                                                 sampler=val_sampler)

    max_iou = 0.
    filename = 'drnet.pth'

    for epoch in range(args.start_epoch, args.epochs):
        if args.fix_random_seed_val:
            torch.cuda.manual_seed(args.manual_seed + epoch)
            np.random.seed(args.manual_seed + epoch)
            torch.manual_seed(args.manual_seed + epoch)
            torch.cuda.manual_seed_all(args.manual_seed + epoch)
            random.seed(args.manual_seed + epoch)

        if args.distributed:
            train_sampler.set_epoch(epoch)    

        epoch_log = epoch + 1
        loss_train, aux_loss_train, mIoU_train, mAcc_train, allAcc_train = train(train_loader, model, optimizer, epoch)
        if main_process():
            writer.add_scalar('TRAIN/loss', loss_train, epoch_log)
            writer.add_scalar('TRAIN/aux_loss', aux_loss_train, epoch_log)
            writer.add_scalar('TRAIN/mIoU', mIoU_train, epoch_log)
            writer.add_scalar('TRAIN/mAcc', mAcc_train, epoch_log)
            writer.add_scalar('TRAIN/allAcc', allAcc_train, epoch_log)     

        if args.evaluate and (epoch % 2 == 0 or (args.epochs<=50 and epoch%1==0)):
            with torch.no_grad():
                loss_val, class_miou = validate(val_loader, model, criterion)
            if main_process():
                writer.add_scalar('VAL/loss', loss_val, epoch_log)
                writer.add_scalar('VAL/class_miou', class_miou, epoch_log)
            if class_miou > max_iou:
                max_iou = class_miou
                if os.path.exists(filename):
                    os.remove(filename)            
                filename = args.save_path + '/train_epoch_' + str(epoch) + '_'+str(max_iou)+'.pth'
                logger.info('Saving checkpoint to: ' + filename)
                torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)

    filename = args.save_path + '/final.pth'
    logger.info('Saving checkpoint to: ' + filename)
    torch.save({'epoch': args.epochs, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)                


def train(train_loader, model, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    main_loss_meter = AverageMeter()
    aux_loss_meter = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.train()

    end = time.time()
    max_iter = args.epochs * len(train_loader)

    print('Warmup: {}'.format(args.warmup))
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        current_iter = epoch * len(train_loader) + i + 1
        index_split = -1
        if args.base_lr > 1e-6:
            poly_learning_rate(optimizer, args.base_lr, current_iter, max_iter, power=args.power, index_split=index_split, warmup=args.warmup, warmup_step=len(train_loader)//2)

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        
        output, main_loss = model(x=input, y=target)
        loss = main_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n = input.size(0) # batch_size
        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        main_loss_meter.update(main_loss.item(), n)
        loss_meter.update(loss.item(), n)

        batch_time.update(time.time() - end)
        end = time.time()

        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'MainLoss {main_loss_meter.val:.4f} '
                        'AuxLoss {aux_loss_meter.val:.4f} '                        
                        'Loss {loss_meter.val:.4f} '
                        'Accuracy {accuracy:.4f}.'.format(epoch+1, args.epochs, i + 1, len(train_loader),
                                                          batch_time=batch_time,
                                                          data_time=data_time,
                                                          remain_time=remain_time,
                                                          main_loss_meter=main_loss_meter,
                                                          aux_loss_meter=aux_loss_meter,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))
        if main_process():
            writer.add_scalar('TRAIN-BATCH/loss', main_loss_meter.val, current_iter)
            writer.add_scalar('TRAIN-BATCH/aux_loss', aux_loss_meter.val, current_iter)
            writer.add_scalar('TRAIN-BATCH/mIoU', np.mean(intersection / (union + 1e-10)), current_iter)
            writer.add_scalar('TRAIN-BATCH/mAcc', np.mean(intersection / (target + 1e-10)), current_iter)
            writer.add_scalar('TRAIN-BATCH/allAcc', accuracy, current_iter)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    if main_process():
        logger.info('Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch, args.epochs, mIoU, mAcc, allAcc))
        for i in range(args.classes):
            logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
       
    return main_loss_meter.avg, aux_loss_meter.avg, mIoU, mAcc, allAcc


def validate(val_loader, model, criterion):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')

    batch_time = AverageMeter()
    model_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    
    class_intersection_meter = [0]*(args.classes-1)
    class_union_meter = [0]*(args.classes-1)

    if args.manual_seed is not None and args.fix_random_seed_val:
        torch.cuda.manual_seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        random.seed(args.manual_seed)

    model.eval()
    end = time.time()
    val_start = end

    iter_num = 0

    for i, (input, target, ori_label) in enumerate(val_loader):  
        data_time.update(time.time() - end)

        iter_num += 1

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        ori_label = ori_label.cuda(non_blocking=True)

        start_time = time.time()
        output = model(x=input, y=target)
        model_time.update(time.time() - start_time)

        if args.ori_resize:
            longerside = max(ori_label.size(1), ori_label.size(2))
            backmask = torch.ones(ori_label.size(0), longerside, longerside).cuda()*255
            backmask[0, :ori_label.size(1), :ori_label.size(2)] = ori_label
            target = backmask.clone().long()

        output = F.interpolate(output, size=target.size()[1:], mode='bilinear', align_corners=True)
        loss = criterion(output, target)

        output = output.max(1)[1]

        intersection, union, new_target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        intersection, union, target, new_target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy(), new_target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(new_target)
            
        for idx in range(1, len(intersection)):
            class_intersection_meter[idx - 1] += intersection[idx]
            class_union_meter[idx - 1] += union[idx]

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if ((iter_num % 100 == 0) or (iter_num == len(val_loader))) and main_process():
            logger.info('Test: [{}/{}] '
                            'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                            'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                            'Accuracy {accuracy:.4f}.'.format(iter_num, len(val_loader),
                                                            data_time=data_time,
                                                            batch_time=batch_time,
                                                            loss_meter=loss_meter,
                                                            accuracy=accuracy))
    val_time = time.time() - val_start
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    
    class_iou_class = []
    class_miou = 0
    for i in range(len(class_intersection_meter)):
        class_iou = class_intersection_meter[i]/(class_union_meter[i]+ 1e-10)
        class_iou_class.append(class_iou)
        class_miou += class_iou
    class_miou = class_miou*1.0 / len(class_intersection_meter)

    if main_process():
        logger.info('meanIoU---Val result: mIoU {:.4f}.'.format(class_miou)) 
        for i in range(len(class_intersection_meter)):
            logger.info('Class_{} Result: iou_b {:.4f}.'.format(i+1, class_iou_class[i]))   
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
        print('total time: {:.4f}, avg inference time: {:.4f}, count: {}'.format(val_time, model_time.avg, iter_num))

    return loss_meter.avg, class_miou


if __name__ == '__main__':
    main()
