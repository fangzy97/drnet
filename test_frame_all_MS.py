import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
from matplotlib import pyplot as plt
import cv2
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import json
import shutil
import argparse
from tqdm import tqdm
import my_optim
from ss_datalayer import SSDatalayer
from oneshot import *
from utils.Restore import restore
from utils import AverageMeter
from utils.save_atten import SAVE_ATTEN
from utils.segscorer import SegScorer
import seaborn as sns
import matplotlib.patheffects as PathEffects
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from utils import Metrics

ROOT_DIR = '/'.join(os.getcwd().split('/'))
print ROOT_DIR

save_atten = SAVE_ATTEN()

SNAPSHOT_DIR = os.path.join(ROOT_DIR, 'snapshots')
# SNAPSHOT_DIR = os.path.join(ROOT_DIR, 'snapshots_mlcls')
DISP_INTERVAL = 20
RS = 20150101
# Colour map.
label_colours = [(0,0,0)
                # 0=background
                ,(128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128)
                # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                ,(0,128,128),(128,128,128),(64,0,0),(192,0,0),(64,128,0)
                # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                ,(192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128)
                # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
                ,(0,64,0),(128,64,0),(0,192,0),(128,192,0),(0,64,128)]
                # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor

color = ['#800000','#008000','#808000','#000080','#800080',
         '#008080','#808080','#400000','#C00000','#408000',
         '#C08000','#400080','#C00080','#408080','#C08080',
         '#004000','#804000','#00C000','#80C000','#004080']

def get_arguments():
    parser = argparse.ArgumentParser(description='OneShot')
    parser.add_argument("--arch", type=str,default='drnet')
    parser.add_argument("--disp_interval", type=int, default=100)
    parser.add_argument("--snapshot_dir", type=str, default=SNAPSHOT_DIR)

    # parser.add_argument("--group", type=int, default=0)
    parser.add_argument('--num_folds', type=int, default=4)
    parser.add_argument('--restore_step', type=int, default=30000)
    parser.add_argument('--store_image', type=bool, default=False)

    return parser.parse_args()

def plot_embedding(data, label, title, group):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()
    fig, ax = plt.subplots()

    for i in range(group*5,group*5+5):
        x = []
        y = []
        for j in range(data.shape[0]):
            if int(label[j]) == i:
                x.append(data[j, 0])
                y.append(data[j, 1])
        x = np.array(x)
        y = np.array(y)
        plt.scatter(x,y, c=color[i], label=str(i))
    # if group<2:
    #     for i in range(data.shape[0]):
    #         plt.text(data[i, 0], data[i, 1], str(label[i]), color=plt.cm.Set1(label[i]),fontdict={'weight': 'bold', 'size': 7})
    # else:
    #     for i in range(data.shape[0]):
    #         plt.text(data[i, 0], data[i, 1], str(label[i]), color=plt.cm.Set3(label[i]-10),fontdict={'weight': 'bold', 'size': 7})

    plt.legend(loc='upper right')
    #ax.grid(True)
    return fig

def measure(y_in, pred_in):
    # thresh = .5
    thresh = .5
    y = y_in>thresh
    pred = pred_in>thresh
    tp = np.logical_and(y,pred).sum()
    tn = np.logical_and(np.logical_not(y), np.logical_not(pred)).sum()
    fp = np.logical_and(np.logical_not(y), pred).sum()
    fn = np.logical_and(y, np.logical_not(pred)).sum()
    return tp, tn, fp, fn

def restore(args, model, group):
    savedir = os.path.join(args.snapshot_dir, args.arch, 'group_%d_of_%d'%(group, args.num_folds))
    filename='step_%d.pth.tar'%(args.restore_step)
    snapshot = os.path.join(savedir, filename)
    assert os.path.exists(snapshot), "Snapshot file %s does not exist."%(snapshot)

    checkpoint = torch.load(snapshot)
    model.load_state_dict(checkpoint['state_dict'])

    print('Loaded weights from %s'%(snapshot))

def get_model(args):
    model = eval(args.arch).OneModel(args)
    # model = nn.DataParallel(model, device_ids=[4, 5, 6])
    model = model.cuda()

    return model

def draw_mask(mask, image, category):
    # mask1 = np.sum(mask, axis=2, keepdims=True)
    # mask1 = mask1 / 765
    pixels = np.ones((505, 505, 3), dtype=np.uint8)
    pixels = pixels * mask.reshape((505,505,1)) * label_colours[category+1]
    img = cv2.addWeighted(image.astype(np.uint8), 0.7, pixels.astype(np.uint8), 0.9, 0)

    return img
def get_org_img(img):
    # print(np.max(img))
    img = np.transpose(img, (1,2,0))
    mean_vals = [0.485, 0.456, 0.406]
    std_vals = [0.229, 0.224, 0.225]
    img = img*std_vals + mean_vals
    img = img*255
    #img = np.transpose(img, (1, 2, 0))
    return img

def val(args):

    model = get_model(args)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    num_classes = 20
    tp_list = [0]*num_classes
    fp_list = [0]*num_classes
    fn_list = [0]*num_classes
    iou_list = [0]*num_classes

    hist = np.zeros((21, 21))

    a_hist = np.zeros((21, 21))
    scorer = SegScorer(num_classes=21)

    for group in range(4):
        label_list = []
        feat_list = []
        a_hist = np.zeros((21, 21))
        datalayer = SSDatalayer(group)
        restore(args, model, group)

        for count in tqdm(range(1000)):
            dat = datalayer.dequeue()
            query_img = dat['second_img'][0]
            ref_img = dat['first_img'][0]
            ref_label = dat['second_label'][0]
            query_label = dat['first_label'][0]
            deploy_info = dat['deploy_info']

            semantic_label = deploy_info['first_semantic_labels'][0][0]

            ref_img, ref_label = torch.Tensor(ref_img).cuda(), torch.Tensor(ref_label).cuda()
            query_img, query_label = torch.Tensor(query_img).cuda(), torch.Tensor(query_label[0,:,:]).cuda()

            ref_img_var, query_img_var = Variable(ref_img), Variable(query_img)
            query_label_var, ref_label_var = Variable(query_label), Variable(ref_label)

            ref_img_var = torch.unsqueeze(ref_img_var,dim=0)
            ref_label_var = torch.unsqueeze(ref_label_var, dim=1)
            query_img_var = torch.unsqueeze(query_img_var, dim=0)
            query_label_var = torch.unsqueeze(query_label_var, dim=0)
            h, w = query_label.size()

            logits1  = model(query_img_var, ref_img_var, ref_label_var,ref_label_var)
            values1, pred1 = model.get_pred(logits1, query_img_var)
            del logits1

            ref_img_var2 = F.upsample(ref_img_var, scale_factor=1.3, mode='bilinear')
            query_img_var2 = F.upsample(query_img_var, scale_factor=1.3, mode='bilinear')
            ref_label_var2 = F.upsample(ref_label_var, scale_factor=1.3, mode='nearest')
            logits2 = model(query_img_var2, ref_img_var2, ref_label_var2, ref_label_var2)
            values2, pred2 = model.get_pred(logits2, query_img_var)
            del logits2
            values2 = F.upsample(values2, (h, w), mode='bilinear')

            ref_img_var3 = F.upsample(ref_img_var, scale_factor=0.7, mode='bilinear')
            query_img_var3 = F.upsample(query_img_var, scale_factor=0.7, mode='bilinear')
            ref_label_var3 = F.upsample(ref_label_var, scale_factor=0.7, mode='nearest')
            logits3 = model(query_img_var3, ref_img_var3, ref_label_var3, ref_label_var3)
            values3, pred3 = model.get_pred(logits3, query_img_var)
            del logits3
            values3 = F.upsample(values3, (h, w), mode='bilinear')

            value = (values1+values2+values3)/3
            value = F.softmax(value, dim=1).squeeze()
            values, pred = torch.max(value, dim=0)

            pred = pred.data.cpu().numpy()
            query_label = query_label.cpu().numpy().astype(np.int32)
            # query_label = cv2.resize(query_label, (505, 505), interpolation=cv2.INTER_NEAREST)
            class_ind = int(deploy_info['first_semantic_labels'][0][0])-1 # because class indices from 1 in data layer
            # label_list.append(class_ind)

            # draw the t-sne picture

            scorer.update(pred, query_label, class_ind+1)
            tp, tn, fp, fn = measure(query_label, pred)
            # iou_img = tp/float(max(tn+fp+fn,1))
            tp_list[class_ind] += tp
            fp_list[class_ind] += fp
            fn_list[class_ind] += fn
            # max in case both pred and label are zero
            iou_list = [tp_list[ic] /
                        float(max(tp_list[ic] + fp_list[ic] + fn_list[ic],1))
                        for ic in range(num_classes)]


            tmp_pred = pred
            tmp_pred[tmp_pred>0.5] = class_ind+1
            tmp_gt_label = query_label
            tmp_gt_label[tmp_gt_label>0.5] = class_ind+1

            hist += Metrics.fast_hist(tmp_pred, query_label, 21)
            a_hist += Metrics.fast_hist(tmp_pred, query_label, 21)

        feat_list = np.array(feat_list)
        label_list = np.array(label_list)
        # digits_proj = TSNE(random_state=RS).fit_transform(feat_list)
        # fig = plot_embedding(digits_proj, label_list,'',group)
        #
        # plt.savefig(str(group)+'.png')
        print("-------------GROUP %d-------------"%(group))
        print iou_list
        class_indexes = range(group*5, (group+1)*5)
        print 'Mean:', np.mean(np.take(iou_list, class_indexes))
        binary_hist = np.array((a_hist[0, 0], a_hist[0, 1:].sum(), a_hist[1:, 0].sum(), a_hist[1:, 1:].sum())).reshape((2, 2))
        bin_iu = np.diag(binary_hist) / (binary_hist.sum(1) + binary_hist.sum(0) - np.diag(binary_hist))
        print('Bin_iu:', bin_iu)

    print('BMVC IOU', np.mean(np.take(iou_list, range(0,20))))

    miou = Metrics.get_voc_iou(hist)
    print('IOU:', miou, np.mean(miou))


    binary_hist = np.array((hist[0, 0], hist[0, 1:].sum(),hist[1:, 0].sum(), hist[1:, 1:].sum())).reshape((2, 2))
    bin_iu = np.diag(binary_hist) / (binary_hist.sum(1) + binary_hist.sum(0) - np.diag(binary_hist))
    print('Bin_iu:', bin_iu)

    # scores = scorer.score()
    # for k in scores.keys():
    #     print(k, np.mean(scores[k]), scores[k])

if __name__ == '__main__':
    args = get_arguments()
    print 'Running parameters:\n'
    print json.dumps(vars(args), indent=4, separators=(',', ':'))
    if not os.path.exists(args.snapshot_dir):
        os.mkdir(args.snapshot_dir)
    val(args)
