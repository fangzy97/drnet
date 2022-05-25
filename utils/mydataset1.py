from __future__ import print_function
from __future__ import absolute_import

from torch.utils.data import Dataset
import numpy as np
import os
import torch
from PIL import Image
import random
# from .transforms import functional
# random.seed(1234)
# from .transforms import functional
import cv2
import math
from .transforms import transforms
from torch.utils.data import DataLoader
from utils.config import cfg
from datasets.factory import get_imdb


class mydataset(Dataset):

    """Face Landmarks dataset."""

    def __init__(self, args, is_train=True, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.img_db = get_imdb('voc_2012_train_hc')
        self.img_db.get_seg_items(args.group, args.num_folds)
        self.transform = transform
        self.split = args.split
        self.group = args.group
        self.num_folds = args.num_folds
        self.is_train = is_train
        self.crop_h, self.crop_w = 321, 321
        self.mean = (128, 128, 128)


    def __len__(self):
        # return len(self.image_list)
        return 100000000

    def read_img(self, path):
        # img = cv2.imread(path)
        # img = cv2.resize(img, (513, 513))
        return cv2.imread(path)

    def _read_data(self, item_dict):
        img_path = item_dict['img_path']
        mask_path = item_dict['mask_path']

        img_dat = self.read_img(img_path)
        mask_dat = self.read_img(mask_path)
        mask_dat[mask_dat>0] = 1
        return img_dat, mask_dat[:,:,0].astype(np.float32)

    def _read_mlclass_val(self, item_dict):
        img_path = item_dict['img_path']
        mask_path = item_dict['mask_path']

        img_dat = self.read_img(img_path)
        mask_dat = self.read_img(mask_path)

        return img_dat, mask_dat[:,:,0].astype(np.float32)

    def _read_mlclass_train(self, item_dict, category):
        img_path = item_dict['img_path']
        mask_path = item_dict['mask_path']

        img_dat = self.read_img(img_path)
        mask_dat = self.read_img(mask_path)
        mask_dat[mask_dat!=category+1] = 0
        mask_dat[mask_dat==category+1] = 1

        return img_dat, mask_dat[:,:,0].astype(np.float32)

    def _read_mlclass_train2(self, item_dict, category):
        img_path = item_dict['img_path']
        mask_path = item_dict['mask_path']
        
        img_dat = self.read_img(img_path)
        mask_dat = self.read_img(mask_path)
        mask_dat[(mask_dat!=category+1)&(mask_dat!=255)] = 0
        mask_dat[mask_dat==category+1] = 1

        return img_dat, mask_dat[:,:,0].astype(np.float32)

    def _read_mlclass_train1(self, item_dict, category):
        img_path = item_dict['img_path']
        mask_path = item_dict['mask_path']

        img_dat = self.read_img(img_path)
        mask_dat = self.read_img(mask_path)
        mask_dat = mask_dat[:,:,0].astype(np.float32)
        mask_dat = mask_dat - 6
        mask_dat[mask_dat<0] = 255
        mask_dat[mask_dat>100] = 255
        return img_dat, mask_dat

    def _read_mlclass_train3(self, item_dict, category):
        img_path = item_dict['img_path']
        mask_path = item_dict['mask_path']

        img_dat = self.read_img(img_path)
        mask_dat = self.read_img(mask_path)
        mask_dat = mask_dat[:,:,0].astype(np.float32)
        mask_dat[mask_dat == 255] = 0
        # mask_dat[(mask_dat!=category+1)&(mask_dat!=0)] = 255
        mask_dat[mask_dat==category+1] = 1
        return img_dat, mask_dat

    def _read_class_train(self, item_dict, category):
        img_path = item_dict['img_path']
        mask_path = item_dict['mask_path']

        img_dat = self.read_img(img_path)
        mask_dat = self.read_img(mask_path)
        mask_dat[mask_dat==255] = 0
        # mask_dat[mask_dat==category+1] = 1

        return img_dat, mask_dat[:,:,0].astype(np.float32)

    def get_item_mlclass_val(self, query_img, sup_img_list):
        que_img, que_mask = self._read_mlclass_val(query_img)
        supp_img = []
        supp_mask = []
        for img_dit in sup_img_list:
            tmp_img, tmp_mask = self._read_mlclass_val(img_dit)
            supp_img.append(tmp_img)
            supp_mask.append(tmp_mask)

        supp_img_processed = []
        if self.transform is not None:
            que_img = self.transform(que_img)
            for img in supp_img:
                supp_img_processed.append(self.transform(img))

        return que_img, que_mask, supp_img_processed, supp_mask
    
    def generate_scale_label(self, image, label):
        f_scale = 0.5 + random.randint(0, 11) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)
        label[label>0] = 1
        image = np.asarray(image, np.float32)
        image -= self.mean
        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                         pad_w, cv2.BORDER_CONSTANT,
                                         value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0,
                                           pad_w, cv2.BORDER_CONSTANT,
                                           value=(0.0))
        else:
            img_pad, label_pad = image, label

        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        # roi = cv2.Rect(w_off, h_off, self.crop_w, self.crop_h);
        image = np.asarray(img_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)

        return image, label
    
    def get_item_mlclass_train(self, query_img, support_img, category):
        que_img, que_mask = self._read_mlclass_train(query_img, category)
        supp_img, supp_mask = self._read_mlclass_train(support_img, category)
        neg_img, neg_mask = self._read_mlclass_train(query_img, category)
        
        if self.transform is not None:
            que_img = Image.fromarray(np.uint8(que_img))
            supp_img = Image.fromarray(np.uint8(supp_img))
            neg_img = Image.fromarray(np.uint8(neg_img))
            que_img = self.transform(que_img)
            supp_img = self.transform(supp_img)
            neg_img = self.transform(neg_img)
            que_mask = cv2.resize(que_mask, (321,321), interpolation=cv2.INTER_NEAREST)
            neg_mask = cv2.resize(neg_mask, (321,321), interpolation=cv2.INTER_NEAREST)
            supp_mask = cv2.resize(supp_mask, (321,321), interpolation=cv2.INTER_NEAREST)
        return que_img, que_mask, supp_img, supp_mask, neg_img, neg_mask, category+1

    def get_item_class_train(self, query_img, support_img, category):
        que_img, que_mask = self._read_class_train(query_img, category)
        supp_img, supp_mask = self._read_class_train(support_img, category)
        neg_img, neg_mask = self._read_data(query_img)

        if self.transform is not None:
            que_img = Image.fromarray(np.uint8(que_img))
            supp_img = Image.fromarray(np.uint8(supp_img))
            neg_img = Image.fromarray(np.uint8(neg_img))
            que_img = self.transform(que_img)
            supp_img = self.transform(supp_img)
            neg_img = self.transform(neg_img)
            que_mask = cv2.resize(que_mask, (321,321), interpolation=cv2.INTER_NEAREST)
            neg_mask = cv2.resize(neg_mask, (321,321), interpolation=cv2.INTER_NEAREST)
            supp_mask = cv2.resize(supp_mask, (321,321), interpolation=cv2.INTER_NEAREST)
        return que_img, que_mask, supp_img, supp_mask, neg_img, neg_mask, category+1

    def get_item_single_train(self,dat_dicts):
        first_img, first_mask = self._read_data(dat_dicts[0])
        second_img, second_mask = self._read_data(dat_dicts[1])
        thrid_img, thrid_mask = self._read_data(dat_dicts[2])

        if self.transform is not None:
            first_img = self.transform(first_img)
            second_img = self.transform(second_img)
            thrid_img = self.transform(thrid_img)

        return first_img, first_mask, second_img,second_mask, thrid_img, thrid_mask

    def get_item_rand_val(self,dat_dicts):
        first_img, first_mask = self._read_data(dat_dicts[0])
        second_img, second_mask = self._read_data(dat_dicts[1])
        thrid_img, thrid_mask = self._read_data(dat_dicts[2])
        category = dat_dicts[3]
        if self.transform is not None:
            first_img = self.transform(first_img)
            second_img = self.transform(second_img)
            thrid_img = self.transform(thrid_img)

        # return first_img, first_mask, second_img,second_mask
        return first_img, first_mask, second_img,second_mask, thrid_img, thrid_mask, category

    def __getitem__(self, idx):
        if self.split == 'train':
            dat_dicts = self.img_db.get_triple_images(split='train', group=self.group, num_folds=4)
            return self.get_item_single_train(dat_dicts)
        elif self.split == 'random_val':
            dat_dicts = self.img_db.get_triple_images(split='val', group=self.group, num_folds=4)
            return self.get_item_rand_val(dat_dicts)
        elif self.split == 'mlclass_val':
            query_img, sup_img_list = self.img_db.get_multiclass_val(split='val', group=self.group, num_folds=4)
            return self.get_item_mlclass_val(query_img, sup_img_list)
        elif self.split == 'mlclass_train':
            query_img, support_img, category = self.img_db.get_multiclass_train(split='train', group=self.group, num_folds=4)
            return self.get_item_mlclass_train(query_img, support_img, category)
        elif self.split == 'class_train':
            query_img, support_img, category = self.img_db.get_multiclass_train(split='train', group=self.group,
                                                                                num_folds=4)
            return self.get_item_class_train(query_img, support_img, category)


    # def __getitem__(self, idx):
    #     if self.split == 'train':
    #         dat_dicts = self.img_db.get_triple_images(split='train', group=self.group, num_folds=4)
    #
    #     first_img, first_mask = self._read_data(dat_dicts[0])
    #     second_img, second_mask = self._read_data(dat_dicts[1])
    #     thrid_img, thrid_mask = self._read_data(dat_dicts[2])
    #
    #     if self.transform is not None:
    #         first_img = self.transform(first_img)
    #         second_img = self.transform(second_img)
    #         thrid_img = self.transform(thrid_img)
    #
    #     return first_img, first_mask, second_img,second_mask, thrid_img, thrid_mask


