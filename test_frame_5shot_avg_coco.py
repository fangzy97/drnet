import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from tqdm import tqdm
import torch.backends.cudnn as cudnn

from datasets.coco.customized import coco_fewshot
from datasets.coco.transforms import Resize, ToTensorNormalize
from utils import AverageMeter
from utils import Metrics
from utils.save_atten import SAVE_ATTEN
from utils.segscorer import SegScorer
from utils.utils import CLASS_LABELS
from oneshot import *

def setup_seed(seed):
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    print("Set the random seed as {}".format(seed))

# setup_seed(321)

ROOT_DIR = '/'.join(os.getcwd().split('/'))
print(ROOT_DIR)

save_atten = SAVE_ATTEN()
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
# SNAPSHOT_DIR = os.path.join(ROOT_DIR, 'snapshots')
SNAPSHOT_DIR = os.path.join(ROOT_DIR, 'snapshots', 'coco')
DISP_INTERVAL = 80


def get_arguments():
    parser = argparse.ArgumentParser(description='OneShot')
    parser.add_argument("--arch", type=str, default='drnet_1')
    parser.add_argument("--img_size", type=int, default=321)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--disp_interval", type=int, default=100)
    parser.add_argument("--snapshot_dir", type=str, default=SNAPSHOT_DIR)

    # parser.add_argument("--group", type=int, default=0)
    parser.add_argument('--num_folds', type=int, default=4)
    parser.add_argument('--restore_step', type=int, default=30000)

    return parser.parse_args()


def measure(y_in, pred_in):
    # thresh = .5
    thresh = .5
    y = y_in > thresh
    pred = pred_in > thresh
    tp = np.logical_and(y, pred).sum()
    tn = np.logical_and(np.logical_not(y), np.logical_not(pred)).sum()
    fp = np.logical_and(np.logical_not(y), pred).sum()
    fn = np.logical_and(y, np.logical_not(pred)).sum()
    return tp, tn, fp, fn


def restore(args, model, group):
    savedir = os.path.join(
        args.snapshot_dir,
        args.arch + '_batch_' + str(args.batch_size) + '_' + str(args.img_size),
        'group_%d_of_%d' % (group, args.num_folds)
    )
    filename = 'step_%d.pth.tar' % (args.restore_step)
    snapshot = os.path.join(savedir, filename)
    assert os.path.exists(snapshot), "Snapshot file %s does not exist." % (snapshot)

    checkpoint = torch.load(snapshot)
    model.load_state_dict(checkpoint['state_dict'])

    print('Loaded weights from %s' % (snapshot))


def get_model(args):
    model = eval(args.arch).OneModel(args)

    model = model.cuda()

    return model


def val(args):
    losses = AverageMeter()
    model = get_model(args)
    model.eval()

    num_classes = 80
    tp_list = [0] * num_classes
    fp_list = [0] * num_classes
    fn_list = [0] * num_classes
    iou_list = [0] * num_classes

    hist = np.zeros((81, 81))
    scorer = SegScorer(num_classes=81)

    transforms = Compose([Resize(size=(args.img_size, args.img_size))])

    new_data = {}
    bin_iou = 0.
    for group in range(4):
        restore(args, model, group)

        labels = CLASS_LABELS['COCO']['all'] - CLASS_LABELS['COCO'][group]
        coco = coco_fewshot(
            base_dir='/home/zhiyuan/dataset/coco',
            split='val',
            transforms=transforms,
            to_tensor=ToTensorNormalize(),
            labels=labels,
            max_iters=1000,
            n_ways=1,
            n_shots=5,
            n_queries=1
        )
        coco_cls_ids = coco.datasets[0].dataset.coco.getCatIds()
        test_loader = DataLoader(
            coco,
            batch_size=1,
            num_workers=1
        )

        a_hist = np.zeros((81, 81))
        for dat in tqdm(test_loader):
            label_ids = [coco_cls_ids.index(x) + 1 for x in dat['class_ids']]

            sup_img = torch.cat([torch.cat([shot.cuda() for shot in way]) for way in dat['support_images']])
            sup_label = torch.cat(
                [torch.cat([shot['fg_mask'].float().cuda() for shot in way]) for way in dat['support_mask']])

            que_img = torch.cat([que.cuda() for que in dat['query_images']])
            que_label = torch.cat([que_label.cuda() for que_label in dat['query_labels']])

            sup_img = [img for img in sup_img]
            que_img, que_label = que_img[0], que_label[0]
            que_img = que_img.unsqueeze(0)

            for p in model.parameters():
                p.requires_grad = False
            logits = model.forward_5shot_avg(que_img, sup_img, sup_label)

            values, pred, coarse = model.get_pred(logits, que_img)
            pred = pred.data.cpu().numpy()

            que_label = que_label.cpu().numpy()
            class_ind = label_ids[0] - 1

            scorer.update(pred, que_label, class_ind + 1)
            tp, tn, fp, fn = measure(que_label, pred)
            tp_list[class_ind] += tp
            fp_list[class_ind] += fp
            fn_list[class_ind] += fn
            # max in case both pred and label are zero
            iou_list = [tp_list[ic] /
                        float(max(tp_list[ic] + fp_list[ic] + fn_list[ic], 1))
                        for ic in range(num_classes)]

            tmp_pred = pred
            tmp_pred[tmp_pred > 0.5] = class_ind + 1
            tmp_gt_label = que_label
            tmp_gt_label[tmp_gt_label > 0.5] = class_ind + 1

            hist += Metrics.fast_hist(tmp_pred, que_label, 81)
            a_hist += Metrics.fast_hist(tmp_pred, que_label, 81)

        print("-------------GROUP %d-------------" % (group))
        print(iou_list)
        class_indexes = range(group * 20, (group + 1) * 20)
        miou = np.mean(np.take(iou_list, class_indexes))
        new_data['split-{}'.format(group)] = round(miou * 100, 1)
        print('Mean:', miou)

        binary_hist = np.array((a_hist[0, 0], a_hist[0, 1:].sum(), a_hist[1:, 0].sum(), a_hist[1:, 1:].sum())).reshape(
            (2, 2))
        bin_iu = np.diag(binary_hist) / (binary_hist.sum(1) + binary_hist.sum(0) - np.diag(binary_hist))
        print('Bin_iu:', bin_iu)
        bin_iou += sum(bin_iu) / len(bin_iu)

    bmvc_iou = np.mean(np.take(iou_list, range(0, 80)))
    new_data['mean'] = round(bmvc_iou * 100, 1)
    print('BMVC IOU', bmvc_iou)

    new_data['b-iou'] = round(bin_iou / 4 * 100, 1)

    miou = Metrics.get_voc_iou(hist)
    print('IOU:', miou, np.mean(miou))

    df.loc[len(df)] = new_data
    df.to_csv(log_file)

    scores = scorer.score()
    for k in scores.keys():
        print(k, np.mean(scores[k]), scores[k])


if __name__ == '__main__':
    args = get_arguments()
    print('Running parameters:\n')
    print(json.dumps(vars(args), indent=4, separators=(',', ':')))
    if not os.path.exists(args.snapshot_dir):
        os.mkdir(args.snapshot_dir)

    log_file = os.path.join(
        'test_log',
        '5-shot',
        '{}_batch{}_{}.csv'.format(args.arch, args.batch_size, args.img_size)
    )
    if os.path.exists(log_file):
        df = pd.read_csv(log_file, index_col=0)
    else:
        df = pd.DataFrame(columns=['split-0', 'split-1', 'split-2', 'split-3', 'mean', 'b-iou'])

    val(args)
