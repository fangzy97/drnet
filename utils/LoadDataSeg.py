from __future__ import print_function
from __future__ import absolute_import

from torchvision import transforms
from torch.utils.data import DataLoader
from .mydataset import mydataset
from PIL import ImageFilter
import random

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def data_loader(args):
    mean_vals = [0.485, 0.456, 0.406]
    std_vals = [0.229, 0.224, 0.225]

    tsfm_train = transforms.Compose([
                                     transforms.Resize((321,321), interpolation=3),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals)
                                     ])

    tsfm_aug = transforms.Compose([
                                     transforms.Resize((321,321), interpolation=3),
                                    #  transforms.RandomApply([
                                    #    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                                    #  ], p=0.8),
                                    #  transforms.RandomGrayscale(p=0.2),
                                    #  transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                                    #  transforms.RandomHorizontalFlip(p=1),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals)
                                     ])
    img_train = mydataset(args, transform=tsfm_train, aug=tsfm_aug)

    train_loader = DataLoader(img_train, batch_size=4, shuffle=True, num_workers=1)

    return train_loader

def val_loader(args):
    mean_vals = [0.485, 0.456, 0.406]
    std_vals = [0.229, 0.224, 0.225]

    tsfm_val = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals)
                                     ])


    img_val = mydataset(args, is_train=False, transform=tsfm_val)

    val_loader = DataLoader(img_val, batch_size=1, shuffle=False, num_workers=1)

    return val_loader
