import argparse
import json
import os
import random

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from numpy import ndarray
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from tqdm import tqdm
import seaborn as sns
import torch.backends.cudnn as cudnn

from datasets.coco.customized import coco_fewshot
from datasets.coco.transforms import Resize, ToTensorNormalize
from utils import Metrics
from utils.save_atten import SAVE_ATTEN
from utils.utils import CLASS_LABELS
from oneshot import *

if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
# os.environ["CUDA_VISIBLE_DEVICES"] = "9"

def setup_seed(seed):
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    print("Set the random seed as {}".format(seed))

setup_seed(321)

ROOT_DIR = '/'.join(os.getcwd().split('/'))
print(ROOT_DIR)

save_atten = SAVE_ATTEN()

SNAPSHOT_DIR = os.path.join(ROOT_DIR, 'snapshots', 'coco')

label_colours = [(0, 0, 0),
                 (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                 (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                 (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                 (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128),
                 (102,204,255), (255,51,102), (153,51,153), (153,0,51), (102,102,255),
                 (255,51,0), (0,102,102), (0,204,102), (102,204,153), (238,136,51),
                 (204,153,255), (204,204,0), (102,153,255), (204,0,102), (204,0,153)
                 ]

color = ['#800000', '#008000', '#808000', '#000080', '#800080',
         '#008080', '#808080', '#400000', '#C00000', '#408000',
         '#C08000', '#400080', '#C00080', '#408080', '#C08080',
         '#004000', '#804000', '#00C000', '#80C000', '#004080',
         '#CCCC00', '#660099', '#666600', '#3300CC', '#CCCC99',
         '#66CCFF', '#FF3366', '#993399', '#990033', '#6666FF',
         '#FF3300', '#006666', '#00CC66', '#66CC99', '#EE8833',
         '#CC99FF', '#CCCC00', '#6699FF', '#CC0066', '#CC0099']


def get_arguments():
    parser = argparse.ArgumentParser(description='OneShot')
    parser.add_argument("--arch", type=str, default='drnet_1')
    parser.add_argument("--img_size", type=int, default=321)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--disp_interval", type=int, default=100)
    parser.add_argument("--snapshot_dir", type=str, default=SNAPSHOT_DIR)
    parser.add_argument('--num_folds', type=int, default=4)
    parser.add_argument('--restore_step', type=int, default=30000)
    parser.add_argument('--store_image', type=bool, default=False)

    return parser.parse_args()


def plot_embedding(data, label, title, group):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig, ax = plt.subplots()

    for i in range(group * 5, group * 5 + 5):
        x = []
        y = []
        for j in range(data.shape[0]):
            if int(label[j]) == i:
                x.append(data[j, 0])
                y.append(data[j, 1])
        x = np.array(x)
        y = np.array(y)
        plt.scatter(x, y, c=color[i], label=str(i))
    # if group<2:
    #     for i in range(data.shape[0]):
    #         plt.text(data[i, 0], data[i, 1], str(label[i]), color=plt.cm.Set1(label[i]),fontdict={'weight': 'bold', 'size': 7})
    # else:
    #     for i in range(data.shape[0]):
    #         plt.text(data[i, 0], data[i, 1], str(label[i]), color=plt.cm.Set3(label[i]-10),fontdict={'weight': 'bold', 'size': 7})

    plt.legend(loc='upper right')
    # ax.grid(True)
    return fig


def draw_mask(args, mask, image, category):
    mask = mask * 255
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    pixels = mask / 255 * label_colours[category + 1]
    pixels = pixels.astype('uint8')

    img = cv2.addWeighted(image, 0.7, pixels, 0.9, 0)

    return img


def get_org_img(img):
    # print(np.max(img))
    img = img.cpu().numpy()
    img = img.squeeze()
    img = np.transpose(img, (1, 2, 0))
    mean_vals = [0.485, 0.456, 0.406]
    std_vals = [0.229, 0.224, 0.225]
    img = img * std_vals + mean_vals
    img = img * 255
    img = img.astype('uint8')
    # img = np.transpose(img, (1, 2, 0))
    return img


def measure(y_in, pred_in):
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
        # args.arch,
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


def denormalize(x):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    x = x.transpose((1, 2, 0))
    out = x * std + mean
    out = out * 255
    return out


def fuse(src, dst):
    color = [0, 0, 128]
    src = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
    src = src / 255 * color
    out = cv2.addWeighted(dst, 1, src, 0.9, 0)

    return out


def val(args):
    # set_seed(1234)

    model = get_model(args)
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    num_classes = 80
    tp_list = [0] * num_classes
    fp_list = [0] * num_classes
    fn_list = [0] * num_classes
    iou_list = [0] * num_classes

    hist = np.zeros((81, 81))

    transforms = Compose([Resize(size=(args.img_size, args.img_size))])

    new_data = {}
    count = 0
    for group in range(4):
        a_hist = np.zeros((81, 81))
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
            n_shots=1,
            n_queries=1
        )
        coco_cls_ids = coco.datasets[0].dataset.coco.getCatIds()
        test_loader = DataLoader(
            coco,
            batch_size=1,
            num_workers=1
        )

        for dat in tqdm(test_loader):
            label_ids = [coco_cls_ids.index(x) + 1 for x in dat['class_ids']]

            sup_img = torch.cat([torch.cat([shot.cuda() for shot in way]) for way in dat['support_images']])
            sup_labels = torch.cat(
                [torch.cat([shot['fg_mask'].float().cuda() for shot in way]) for way in dat['support_mask']])
            sup_labels = sup_labels.unsqueeze(0)

            que_img = torch.cat([que.cuda() for que in dat['query_images']])
            que_labels = torch.cat([que_label.cuda() for que_label in dat['query_labels']])
            que_labels = que_labels.unsqueeze(0)

            logits = model(que_img, sup_img, sup_labels, sup_labels)

            if args.store_image:
                sim1 = logits[0]
                sim1 = F.upsample(sim1, (args.img_size, args.img_size), mode='bilinear', align_corners=True)
                sim2 = logits[1]
                sim2 = F.upsample(sim2, (args.img_size, args.img_size), mode='bilinear', align_corners=True)
                sim1 = sim1.squeeze(0)
                sim1 = sim1.squeeze(0)
                sim2 = sim2.squeeze(0)
                sim2 = sim2.squeeze(0)

                sim1 = sim1.cpu().detach().numpy().astype(np.float32)
                ax1 = sns.heatmap(sim1, center=0.5, yticklabels=False, xticklabels=False, square=True)
                ax1.figure.savefig('./output3/fast_' + str(count) + '.png')
                sim2 = sim2.cpu().detach().numpy().astype(np.float32)
                ax2 = sns.heatmap(sim2, center=0.5, yticklabels=False, xticklabels=False, square=True)
                ax2.figure.savefig('./output3/slow_' + str(count) + '.png')

            values, pred, coarse = model.get_pred(logits, que_labels)

            pred = pred.data.cpu().numpy().astype('uint8')
            coarse = coarse.data.cpu().numpy().astype('uint8')

            que_labels = que_labels.cpu().numpy().astype('uint8').squeeze()
            # because class indices from 1 in data layer
            class_ind = label_ids[0] - 1

            tp, tn, fp, fn = measure(que_labels, pred)

            tp_list[class_ind] += tp
            fp_list[class_ind] += fp
            fn_list[class_ind] += fn

            # max in case both pred and label are zero
            iou_list = [tp_list[ic] / float(max(tp_list[ic] + fp_list[ic] + fn_list[ic], 1)) for ic in
                        range(num_classes)]

            tmp_pred = pred
            tmp_pred[tmp_pred > 0.5] = class_ind + 1
            tmp_gt_label = que_labels
            tmp_gt_label[tmp_gt_label > 0.5] = class_ind + 1
            hist += Metrics.fast_hist(tmp_pred, tmp_gt_label, 81)
            a_hist += Metrics.fast_hist(tmp_pred, tmp_gt_label, 81)

            count += 1

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

    bmvc_iou = np.mean(np.take(iou_list, range(0, 80)))
    new_data['mean'] = round(bmvc_iou * 100, 1)
    print('BMVC IOU', bmvc_iou)

    miou = Metrics.get_voc_iou(hist)
    print('IOU:', miou, np.mean(miou))

    binary_hist = np.array((hist[0, 0], hist[0, 1:].sum(), hist[1:, 0].sum(), hist[1:, 1:].sum())).reshape((2, 2))
    bin_iu = np.diag(binary_hist) / (binary_hist.sum(1) + binary_hist.sum(0) - np.diag(binary_hist))
    print('Bin_iu:', bin_iu)
    new_data['b-iou'] = round(sum(bin_iu) / len(bin_iu) * 100, 1)

    update = pd.DataFrame(new_data, index=[0])
    save_log = df.append(update, ignore_index=True)
    save_log.to_csv(log_file)


if __name__ == '__main__':
    args = get_arguments()
    print('Running parameters:\n')
    print(json.dumps(vars(args), indent=4, separators=(',', ':')))
    if not os.path.exists(args.snapshot_dir):
        os.mkdir(args.snapshot_dir)

    log_file = os.path.join('test_log', '{}_batch{}_{}.csv'.format(args.arch, args.batch_size, args.img_size))
    if os.path.exists(log_file):
        df = pd.read_csv(log_file, index_col=0)
    else:
        df = pd.DataFrame(columns=['split-0', 'split-1', 'split-2', 'split-3', 'mean', 'b-iou'])

    val(args)
