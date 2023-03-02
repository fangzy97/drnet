import os
import os.path
import random
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
import json

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']



def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(split=0, data_root=None, data_list=None, sub_list=None):    
    assert split in [0, 1, 2, 3, 10, 11, 999]
    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))

    split_data_list  = data_list[:data_list.rindex('.')] + f'_split{split}.pth'
    if os.path.isfile(split_data_list):
        print("Load split data list from " + split_data_list)
        image_label_list, sub_class_file_list = torch.load(split_data_list)
        return image_label_list, sub_class_file_list

    # Shaban uses these lines to remove small objects:
    # if util.change_coordinates(mask, 32.0, 0.0).sum() > 2:
    #    filtered_item.append(item)      
    # which means the mask will be downsampled to 1/32 of the original size and the valid area should be larger than 2, 
    # therefore the area in original size should be accordingly larger than 2 * 32 * 32    
    image_label_list = []  
    list_read = open(data_list).readlines()
    print("Processing data...".format(sub_list))
    sub_class_file_list = {}
    for sub_c in sub_list:
        sub_class_file_list[sub_c] = []

    for l_idx in tqdm(range(len(list_read))):
        line = list_read[l_idx]
        line = line.strip()
        line_split = line.split(' ')
        image_name = os.path.join(data_root, line_split[0])
        label_name = os.path.join(data_root, line_split[1])
        item = (image_name, label_name)
        label = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)
        label_class = np.unique(label).tolist()

        # remove 0 and 255 in the label image
        if 0 in label_class:
            label_class.remove(0)
        if 255 in label_class:
            label_class.remove(255)

        # remove those images with small objects
        new_label_class = []       
        for c in label_class:
            if c in sub_list:
                tmp_label = np.zeros_like(label)
                target_pix = np.where(label == c)
                tmp_label[target_pix[0],target_pix[1]] = 1 
                if tmp_label.sum() >= 2 * 32 * 32:      
                    new_label_class.append(c)

        label_class = new_label_class    

        if len(label_class) > 0:
            image_label_list.append(item)
            for c in label_class:
                if c in sub_list:
                    sub_class_file_list[c].append(item)
                    
    print("Checking image&label pair {} list done! ".format(split))
    print("{} images&label pair after processing! ".format(len(image_label_list)))

    print("Saving processed data...")
    torch.save((image_label_list, sub_class_file_list), split_data_list)
    print("Done")

    return image_label_list, sub_class_file_list



class SemData(Dataset):
    def __init__(self, data_path, class_chosen, transform):
        self.class_chosen = class_chosen
        self.transform = transform
        with open(data_path, 'r') as f:
            self.data_list = json.load(f)

    def __len__(self):
        return len(self.data_list)

    def read_img(self, path):
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        image = np.float32(image)

        return image

    def read_label(self, path):
        label = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return label

    def __getitem__(self, index):
        sup_img = self.read_img(self.data_list[index]['sup'])
        sup_label = self.read_label(self.data_list[index]['sup_label'])
        que_img = self.read_img(self.data_list[index]['que'])
        que_label = self.read_label(self.data_list[index]['que_label'])

        target_pix = np.where(sup_label == self.class_chosen)
        ignore_pix = np.where(sup_label == 255)
        sup_label[:,:] = 0
        if target_pix[0].shape[0] > 0:
            sup_label[target_pix[0],target_pix[1]] = 1 
        sup_label[ignore_pix[0],ignore_pix[1]] = 255           

        target_pix = np.where(que_label == self.class_chosen)
        ignore_pix = np.where(que_label == 255)
        que_label[:,:] = 0
        if target_pix[0].shape[0] > 0:
            que_label[target_pix[0],target_pix[1]] = 1 
        que_label[ignore_pix[0],ignore_pix[1]] = 255      

        sup_img, sup_label = self.transform(sup_img, sup_label)
        sup_img = sup_img.unsqueeze(0)
        sup_label = sup_label.unsqueeze(0)
        raw_label = que_label.copy()
        que_img, que_label = self.transform(que_img, que_label)

        return que_img, que_label, sup_img, sup_label, raw_label


class BaseData(Dataset):
    def __init__(self, split=3, mode=None, data_root=None, data_list=None, data_set=None, use_split_coco=False, transform=None, batch_size=None):

        assert data_set in ['pascal', 'coco']
        assert mode in ['train', 'val']

        if data_set == 'pascal':
            self.num_classes = 20
        elif data_set == 'coco':
            self.num_classes = 80

        self.mode = mode
        self.split = split 
        self.data_root = data_root
        self.batch_size = batch_size

        if data_set == 'pascal':
            self.class_list = list(range(1, 21))                         # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
            if self.split == 3: 
                self.sub_list = list(range(1, 16))                       # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
                self.sub_val_list = list(range(16, 21))                  # [16,17,18,19,20]
            elif self.split == 2:
                self.sub_list = list(range(1, 11)) + list(range(16, 21)) # [1,2,3,4,5,6,7,8,9,10,16,17,18,19,20]
                self.sub_val_list = list(range(11, 16))                  # [11,12,13,14,15]
            elif self.split == 1:
                self.sub_list = list(range(1, 6)) + list(range(11, 21))  # [1,2,3,4,5,11,12,13,14,15,16,17,18,19,20]
                self.sub_val_list = list(range(6, 11))                   # [6,7,8,9,10]
            elif self.split == 0:
                self.sub_list = list(range(6, 21))                       # [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
                self.sub_val_list = list(range(1, 6))                    # [1,2,3,4,5]
        elif data_set == 'coco':
            if use_split_coco:
                print('INFO: using SPLIT COCO (FWB)')
                self.class_list = list(range(1, 81))
                if self.split == 3:
                    self.sub_val_list = list(range(4, 81, 4))
                    self.sub_list = list(set(self.class_list) - set(self.sub_val_list))                    
                elif self.split == 2:
                    self.sub_val_list = list(range(3, 80, 4))
                    self.sub_list = list(set(self.class_list) - set(self.sub_val_list))    
                elif self.split == 1:
                    self.sub_val_list = list(range(2, 79, 4))
                    self.sub_list = list(set(self.class_list) - set(self.sub_val_list))    
                elif self.split == 0:
                    self.sub_val_list = list(range(1, 78, 4))
                    self.sub_list = list(set(self.class_list) - set(self.sub_val_list))    
            else:
                print('INFO: using COCO (PANet)')
                self.class_list = list(range(1, 81))
                if self.split == 3:
                    self.sub_list = list(range(1, 61))
                    self.sub_val_list = list(range(61, 81))
                elif self.split == 2:
                    self.sub_list = list(range(1, 41)) + list(range(61, 81))
                    self.sub_val_list = list(range(41, 61))
                elif self.split == 1:
                    self.sub_list = list(range(1, 21)) + list(range(41, 81))
                    self.sub_val_list = list(range(21, 41))
                elif self.split == 0:
                    self.sub_list = list(range(21, 81)) 
                    self.sub_val_list = list(range(1, 21))           

        print('sub_list: ', self.sub_list)
        print('sub_val_list: ', self.sub_val_list)    

        self.data_list = []  
        list_read = open(data_list).readlines()
        print("Processing data...")

        for l_idx in tqdm(range(len(list_read))):
            line = list_read[l_idx]
            line = line.strip()
            line_split = line.split(' ')
            image_name = os.path.join(self.data_root, line_split[0])
            label_name = os.path.join(self.data_root, line_split[1])
            item = (image_name, label_name)
            self.data_list.append(item)

        self.transform = transform


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_path, label_path = self.data_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR) 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label_tmp = label.copy()

        for cls in range(1, self.num_classes+1):
            select_pix = np.where(label_tmp == cls)
            if cls in self.sub_list:
                label[select_pix[0],select_pix[1]] = self.sub_list.index(cls) + 1
            else:
                label[select_pix[0],select_pix[1]] = 0
                
        raw_label = label.copy()

        if self.transform is not None:
            image, label = self.transform(image, label)

        # Return
        if self.mode == 'val':
            return image, label, raw_label

        return image, label