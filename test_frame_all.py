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
import random
import time

# os.environ["CUDA_VISIBLE_DEVICES"] = "9"
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
    parser.add_argument("--bbox", type=bool, default=False)
    parser.add_argument("--scribble", type=bool, default=False)
    parser.add_argument("--image_level", type=bool, default=False)

    # parser.add_argument("--group", type=int, default=0)
    parser.add_argument('--num_folds', type=int, default=4)
    parser.add_argument('--restore_step', type=int, default=30000)
    parser.add_argument('--store_image', type=bool, default=False)
    parser.add_argument('--center', type=float, default=0.0)

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

    model = model.cuda()

    return model

def draw_mask(mask, image, category):
    # mask1 = np.sum(mask, axis=2, keepdims=True)
    # mask1 = mask1 / 765
    pixels = np.ones((505, 505, 3), dtype=np.uint8)
    pixels = pixels * mask.reshape((505,505,1)) * label_colours[(category+1) % len(label_colours)]
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

def get_bbox(fg_mask, inst_mask):
    """
    Get the ground truth bounding boxes
    """

    fg_bbox = torch.zeros_like(fg_mask, device=fg_mask.device)
    bg_bbox = torch.ones_like(fg_mask, device=fg_mask.device)

    inst_mask[fg_mask == 0] = 0
    area = torch.bincount(inst_mask.view(-1))
    cls_id = area[1:].argmax() + 1
    cls_ids = np.unique(inst_mask.cpu().numpy())[1:]

    mask_idx = torch.where(inst_mask[0] == cls_id)
    y_min = mask_idx[0].min()
    y_max = mask_idx[0].max()
    x_min = mask_idx[1].min()
    x_max = mask_idx[1].max()
    fg_bbox[0, y_min:y_max+1, x_min:x_max+1] = 1

    for i in cls_ids:
        mask_idx = torch.where(inst_mask[0] == i)
        y_min = max(mask_idx[0].min(), 0)
        y_max = min(mask_idx[0].max(), fg_mask.shape[1] - 1)
        x_min = max(mask_idx[1].min(), 0)
        x_max = min(mask_idx[1].max(), fg_mask.shape[2] - 1)
        bg_bbox[0, y_min:y_max+1, x_min:x_max+1] = 0
    return fg_bbox, bg_bbox


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

    total_time = 0.
    for group in range(4):
        # group+=1
        label_list = []
        feat_list = []
        a_hist = np.zeros((21, 21))
        datalayer = SSDatalayer(group)
        restore(args, model, group)

        for count in tqdm(range(1000)):
            dat = datalayer.dequeue()
            # ref_img = dat['second_img'][0]
            #cv2.imwrite('hc.png', get_org_img(ref_img))
            # query_img = dat['first_img'][0]
            # query_label = dat['second_label'][0]
            # ref_label = dat['first_label'][0]
            query_img = dat['second_img'][0]
            ref_img = dat['first_img'][0]
            ref_label = dat['second_label'][0]
            query_label = dat['first_label'][0]
            deploy_info = dat['deploy_info']

            # simg = cv2.imread(dat['image1_path'][0])
            # simg = cv2.resize(simg, (505, 505))
            # qimg = cv2.imread(dat['image2_path'][0])
            # qimg = cv2.resize(qimg, (505, 505))
            #
            # cv2.imwrite('./output4/simg' + str(count) + '.png', simg)
            # cv2.imwrite('./output4/qimg' + str(count) + '.png', qimg)
            # qlabel = cv2.resize(query_label.transpose(1, 2, 0), (505, 505))
            # cv2.imwrite('./output4/qmask_' + str(count) + '.png', qlabel * 255)
            # rlabel = cv2.resize(ref_label.transpose(1, 2, 0), (505, 505))
            # cv2.imwrite('./output4/rmask_' + str(count) + '.png', rlabel * 255)
            scribble_label = deploy_info['first_mask_scribble'][0]
            scribble_label = np.where((ref_label == 1)
                              & (scribble_label != 0)
                              & (scribble_label != 255),
                              1, 0)
            scribble_cls_list = list(set(np.unique(scribble_label)) - set([0,]))
            if scribble_cls_list:  # Still need investigation
                scribble_label = np.where(scribble_label == random.choice(scribble_cls_list).item(), 1, 0)
            else:
                scribble_label[:] = 0

            semantic_label = deploy_info['first_semantic_labels'][0][0]
            # hc1 = draw_mask(qlabel, qimg, semantic_label)
            # hc2 = draw_mask(rlabel, simg, semantic_label)
            # cv2.imwrite('./output4/simg' + str(count) + '.png', hc2)
            # cv2.imwrite('./output4/qimg' + str(count) + '.png', hc1)

            ref_img, ref_label, ref_scribble = torch.Tensor(ref_img).cuda(), torch.Tensor(ref_label).cuda(), torch.Tensor(scribble_label).cuda()
            query_img, query_label = torch.Tensor(query_img).cuda(), torch.Tensor(query_label[0,:,:]).cuda()

            # ref_img = ref_img*ref_label
            ref_img_var, query_img_var = Variable(ref_img), Variable(query_img)
            query_label_var, ref_label_var, ref_scribble_var = Variable(query_label), Variable(ref_label), Variable(ref_scribble)

            ref_img_var = torch.unsqueeze(ref_img_var,dim=0)
            ref_label_var = torch.unsqueeze(ref_label_var, dim=1)
            ref_scribble_var = torch.unsqueeze(ref_scribble_var, dim=1)
            # print(torch.max(ref_img_var))
            query_img_var = torch.unsqueeze(query_img_var, dim=0)
            query_label_var = torch.unsqueeze(query_label_var, dim=0)
            # print(torch.max(query_label_var))

            ref_img_var = F.upsample(ref_img_var, (505, 505), mode='bilinear', align_corners=True)
            query_img_var = F.upsample(query_img_var, (505, 505), mode='bilinear', align_corners=True)
            ref_label_var = F.upsample(ref_label_var, (505, 505))
            ref_scribble_var = F.upsample(ref_scribble_var, (505, 505))

            box_mask, _ = get_bbox(ref_label_var, ref_label_var.long())

            # ref_img_var = F.upsample(ref_img_var, (505, 505), mode='bilinear', align_corners=True)
            # print(query_img_var.size())
            start_time = time.time()
            if args.scribble:
                logits = model(query_img_var, ref_img_var, ref_scribble_var, ref_scribble_var)
            elif args.bbox:
                logits = model(query_img_var, ref_img_var, box_mask, box_mask)
            elif args.image_level:
                b, c, h, w = ref_img_var.shape
                mask = torch.ones(b, 1, h, w, device=ref_img_var.device)
                logits = model(query_img_var, ref_img_var, mask, mask)
            else:
                logits = model(query_img_var, ref_img_var, ref_label_var, ref_label_var)
            end_time = time.time()
            total_time += end_time - start_time

            query_img_var1 = query_img_var.squeeze(0)
            query_img_var1 = query_img_var1.cpu().numpy().astype(np.float32)
            # print(np.shape(query_img_var1))

            # print(vec_pos.size())
            # feat_list.append(vec_pos.squeeze(0).data.cpu().numpy())
            if args.store_image:
                center = args.center
                save_path = os.path.join('output_cr', 'thr' + str(center))
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                simg = cv2.imread(dat['image1_path'][0])
                simg = cv2.resize(simg, (505, 505))
                rlabel = cv2.resize(ref_label[0].detach().cpu().numpy(), (505, 505))
                simg = draw_mask(rlabel, simg, semantic_label)
                # cv2.imwrite('{}/{}_{}_simg.png'.format(save_path, group, count), simg)
                qimg = cv2.imread(dat['image2_path'][0])
                qimg = cv2.resize(qimg, (505, 505))
                qlabel = cv2.resize(query_label.detach().cpu().numpy(), (505, 505))
                qimg = draw_mask(qlabel, qimg, semantic_label)
                # cv2.imwrite('{}/{}_{}_qimg.png'.format(save_path, group, count), qimg)

                plt.figure()
                plt.subplot(221)
                plt.imshow(simg)
                plt.axis('off')
                plt.subplot(222)
                plt.imshow(qimg)
                plt.axis('off')

                sim1 = torch.softmax(logits[1], dim=1)
                sim1 = sim1[:, [1]]
                sim1 = F.upsample(sim1, (505, 505), mode='bilinear', align_corners=True)
                sim2 = torch.softmax(logits[2], dim=1)
                sim2 = sim2[:, [1]]
                sim2 = F.upsample(sim2, (505, 505), mode='bilinear', align_corners=True)
                sim1 = sim1.squeeze(0)
                sim1 = sim1.squeeze(0)
                sim2 = sim2.squeeze(0)
                sim2 = sim2.squeeze(0)
                # query_img_var1 = query_img_var.squeeze(0)
                # query_img_var1 = query_img_var1.cpu().numpy().astype(np.float32)
                # query_img_var1 = get_org_img(query_img_var1)
                # cv2.imwrite('./output/'+str(count)+'_q.png', query_img_var1)
                #
                # query_label_var1 = query_label_var
                # query_label_var1 = query_label_var1.cpu().numpy().astype(np.uint8)
                #
                # query_label_var1 = np.transpose(query_label_var1, (1, 2, 0))*255
                # query_label_var1 = cv2.resize(query_label_var1,(505,505))
                # cv2.imwrite('./output/'+str(count)+'_m.png', query_label_var1)

                sim1 = sim1.cpu().detach().numpy().astype(np.float32)
                # f1, ax1 = plt.subplots(figsize=(9, 6))
                plt.subplot(223)
                # plt.figure()
                ax1 = sns.heatmap(sim1, center=center, yticklabels=False, xticklabels=False, vmin=0., vmax=1.)
                # ax1.figure.savefig(os.path.join(save_path, str(group) + '_' + str(count) + '_fast.png'))
                # plt.close()
                # ax1.figure.savefig('./output3/fast_' + str(count) + '.png')

                sim2 = sim2.cpu().detach().numpy().astype(np.float32)
                # f2, ax2 = plt.subplots(figsize=(9, 6))
                plt.subplot(224)
                # plt.figure()
                ax2 = sns.heatmap(sim2, center=center, yticklabels=False, xticklabels=False, vmin=0., vmax=1.)
                # ax2.figure.savefig(os.path.join(save_path, str(group) + '_' + str(count) + '_slow.png'))
                # plt.close()
                # ax2.figure.savefig('./output3/slow_' + str(count) + '.png')

                # sim9 = sim9.cpu().detach().numpy().astype(np.float32)
                # c = sns.heatmap(sim9, center=0, yticklabels=False, xticklabels=False, cbar=False, square=True)
                # c.figure.savefig('./output/' + str(count) + '_9.png')
                save_path_extra = os.path.join(save_path, 'combine')
                if not os.path.exists(save_path_extra):
                    os.makedirs(save_path_extra)
                plt.savefig(os.path.join(save_path_extra, str(group) + '_' + str(count) + '.png'))
                plt.close()


            h, w = query_label.size()
            # outB_side = F.upsample(outB_side, size=(w, h), mode='bilinear')
            # out_side = F.softmax(outB_side, dim=1).squeeze()
            # values, pred = torch.max(out_side, dim=0)
            values, pred = model.get_pred(logits, query_img_var)

            pred = pred.data.cpu().numpy().astype(np.int32)
            # print(np.shape(pred))
            # cv2.imwrite('./output4/com_' + str(count) + '.png', pred * 255)
            # hc3 = draw_mask(pred,qimg,semantic_label)
            # cv2.imwrite('./output4/pred_' + str(count) + '.png', hc3)
            # pred1 = predl[1].data.cpu().numpy().astype(np.int32)
            #
            # hc4 = draw_mask(pred1, qimg, semantic_label)
            # cv2.imwrite('./output4/fast_' + str(count) + '.png', hc4)
            # cv2.imwrite('./output4/com_' + str(count) + '.png', pred * 255)
            pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
            #
            # pred1 = predl[1].data.cpu().numpy().astype(np.int32)
            # cv2.imwrite('./output4/fast_' + str(count) + '.png', pred1 * 255)

            # pred2 = predl[2].data.cpu().numpy().astype(np.int32)
            # cv2.imwrite('./output4/slow_' + str(count) + '.png', pred2 * 255)

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
                        float(max(tp_list[ic] + fp_list[ic] + fn_list[ic], 1))
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

        precision_list = [tp_list[ic] / float(max(tp_list[ic] + fp_list[ic], 1))
                          for ic in range(num_classes)]
        recall_list = [tp_list[ic] / float(max(tp_list[ic] + fn_list[ic], 1))
                          for ic in range(num_classes)]
        print(iou_list)
        class_indexes = range(group*5, (group+1)*5)
        print('Mean:', np.mean(np.take(iou_list, class_indexes)))
        print("Mean precision: {}".format(np.mean(np.take(precision_list, class_indexes))))
        print("Mean recall: {}".format(np.mean(np.take(recall_list, class_indexes))))

        binary_hist = np.array((a_hist[0, 0], a_hist[0, 1:].sum(), a_hist[1:, 0].sum(), a_hist[1:, 1:].sum())).reshape((2, 2))
        bin_iu = np.diag(binary_hist) / (binary_hist.sum(1) + binary_hist.sum(0) - np.diag(binary_hist))
        print('Bin_iu:', bin_iu)

    print('BMVC IOU', np.mean(np.take(iou_list, range(0,20))))

    miou = Metrics.get_voc_iou(hist)
    print('IOU:', miou, np.mean(miou))


    binary_hist = np.array((hist[0, 0], hist[0, 1:].sum(),hist[1:, 0].sum(), hist[1:, 1:].sum())).reshape((2, 2))
    bin_iu = np.diag(binary_hist) / (binary_hist.sum(1) + binary_hist.sum(0) - np.diag(binary_hist))
    print('Bin_iu:', bin_iu)
    print('FPS: ', 4000 / total_time)
    # scores = scorer.score()
    # for k in scores.keys():
    #     print(k, np.mean(scores[k]), scores[k])

if __name__ == '__main__':
    args = get_arguments()
    print('Running parameters:\n')
    print(json.dumps(vars(args), indent=4, separators=(',', ':')))
    if not os.path.exists(args.snapshot_dir):
        os.mkdir(args.snapshot_dir)
    val(args)
