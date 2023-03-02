import torch
import torch.nn.functional as F
from torch import nn

from .lib.ASPP import ASPP
from .lib.non_local import NonLocalBlock2D
from .lib import vgg as vgg_models
from torchvision.models import resnet as models


# Masked Average Pooling
def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat


def get_vgg16_layer(model):
    layer0_idx = range(0,7)
    layer1_idx = range(7,14)
    layer2_idx = range(14,24)
    layer3_idx = range(24,34)
    layer4_idx = range(34,43)
    layers_0 = []
    layers_1 = []
    layers_2 = []
    layers_3 = []
    layers_4 = []
    for idx in layer0_idx:
        layers_0 += [model.features[idx]]
    for idx in layer1_idx:
        layers_1 += [model.features[idx]]
    for idx in layer2_idx:
        layers_2 += [model.features[idx]]
    for idx in layer3_idx:
        layers_3 += [model.features[idx]]
    for idx in layer4_idx:
        layers_4 += [model.features[idx]]  
    layer0 = nn.Sequential(*layers_0) 
    layer1 = nn.Sequential(*layers_1) 
    layer2 = nn.Sequential(*layers_2) 
    layer3 = nn.Sequential(*layers_3) 
    layer4 = nn.Sequential(*layers_4)
    return layer0,layer1,layer2,layer3,layer4


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        layers = args.layers
        classes = args.classes
        assert layers in [50, 101, 152]
        assert classes > 1

        self.zoom_factor = args.zoom_factor
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        self.shot = args.shot
        self.train_iter = args.train_iter
        self.eval_iter = args.eval_iter
        self.vgg = args.vgg

        BatchNorm = nn.BatchNorm2d
        pretrained = True

        if self.vgg:
            print('INFO: Using VGG_16 bn')
            vgg_models.BatchNorm = BatchNorm
            vgg16 = vgg_models.vgg16_bn(pretrained=pretrained)
            self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = get_vgg16_layer(vgg16)
        else:
            print('INFO: Using ResNet {}'.format(layers))
            if layers == 50:
                resnet = models.resnet50(pretrained=pretrained)
            elif layers == 101:
                resnet = models.resnet101(pretrained=pretrained)
            else:
                resnet = models.resnet152(pretrained=pretrained)
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
            self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)

        for param in self.parameters():
            param.requires_grad = False;

        reduce_dim = 256

        # Encoder
        self.side5_1 = nn.Sequential(
            nn.Conv2d(2048 + 1024 + 512, 256, kernel_size=1, padding=0, bias=False),   #fc6
            nn.BatchNorm2d(reduce_dim),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.5)
        )
        self.down_feat = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=1, padding=0, bias=False),  # fc6
            nn.BatchNorm2d(reduce_dim),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.5)
        )
        # CR
        self.non_local = NonLocalBlock2D(reduce_dim)

        # ASPP
        mask_add_num = 1
        self.init_merge = nn.Sequential(
            nn.Conv2d(reduce_dim * 2 + mask_add_num, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True))
        self.aspp = ASPP(reduce_dim)
        self.res1 = nn.Sequential(
            nn.Conv2d(reduce_dim * 5, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(reduce_dim),
            nn.ReLU(inplace=True))
        self.res2 = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(reduce_dim),
            nn.ReLU(inplace=True),   
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(reduce_dim),
            nn.ReLU(inplace=True))
        self.cls = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(reduce_dim),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(p=0.1),
            nn.Conv2d(reduce_dim, 2, kernel_size=1))
        
        
    def freeze_backbone_bn(self):
        for module in self.layer0.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        for module in self.layer1.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        for module in self.layer2.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        for module in self.layer3.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        for module in self.layer4.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    def extract_features(self, x, is_query=False):
        with torch.no_grad():
            x0 = self.layer0(x)
            x1 = self.layer1(x0)
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)
            x4 = self.layer4(x3)
        out = self.down_feat(x4)

        if is_query:
            side_out5_1 = self.side5_1(torch.cat([x4, x3, x2], dim=1))
            return out, side_out5_1

        return out

    def forward(self, x, s_x, s_y, y=None):
        self.freeze_backbone_bn()

        x_size = x.size()
        assert (x_size[2] - 1) % 8 == 0 and (x_size[3] - 1) % 8 == 0
        h = int((x_size[2] - 1) // 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) // 8 * self.zoom_factor + 1)

        # Query Feature
        query_feat, query_feat_side = self.extract_features(x, is_query=True)

        # Support Feature
        supp_feat_list = []
        supp_feat_neg_list = []
        cr_list = []
        for i in range(self.shot):
            mask = (s_y[:, i, :, :] == 1).float().unsqueeze(1)
            mask_neg = (s_y[:, i, :, :] == 0).float().unsqueeze(1)

            supp_feat = self.extract_features(s_x[:, i, :, :, :], is_query=False)
            mask_bin = F.interpolate(mask, size=query_feat.shape[-2:], mode='bilinear', align_corners=True)

            cr_out = self.non_local(query_feat, supp_feat * mask_bin)
            cr_list.append(cr_out)

            supp_feat = F.interpolate(supp_feat, size=mask.shape[-2:], mode='bilinear', align_corners=True)
            supp_feat = Weighted_GAP(supp_feat, mask)
            supp_feat_list.append(supp_feat)

            supp_feat_neg = Weighted_GAP(supp_feat, mask_neg)
            supp_feat_neg_list.append(supp_feat_neg)

        supp_feat = sum(supp_feat_list) / len(supp_feat_list)
        supp_feat_neg = sum(supp_feat_neg_list) / len(supp_feat_neg_list)
        cr_out = sum(cr_list) / len(cr_list)

        # SR module
        fg_init_map = torch.cosine_similarity(query_feat, supp_feat)
        bg_init_map = torch.cosine_similarity(query_feat, supp_feat_neg)
        init_map = torch.stack((bg_init_map, fg_init_map), dim=1) * 20

        init_mask = init_map.argmax(dim=1, keepdim=True)
        rec_vector = Weighted_GAP(query_feat, init_mask.float())
        rec_map = torch.cosine_similarity(query_feat, rec_vector)
        # rec_map = self.normalization(rec_map)
        sr_out = rec_map.unsqueeze(1)

        # fg_init_map = self.normalization(fg_init_map)
        # cr_out = cr_out + query_feat_side * fg_init_map.unsqueeze(1)
        out = self.aspp_pipeline(query_feat_side, sr_out, cr_out)

        #   Output Part
        if self.zoom_factor != 1:
            out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)
            init_map = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)
            fin_out = out + 0.4 * init_map

        if self.training:
            main_loss = self.criterion(out, y.long())  # l_re
            aux_loss = self.criterion(init_map, y.long())  # l_init

            return fin_out.max(1)[1], main_loss, aux_loss * 0.4

        return fin_out

    def aspp_pipeline(self, query_feat, sr_out, cr_out):
        merge_feat = torch.cat([query_feat, sr_out, cr_out], dim=1)
        merge_feat = self.init_merge(merge_feat)

        out = self.aspp(merge_feat)
        out = self.res1(out)   # 1080->256
        out = self.res2(out) + out 
        out = self.cls(out)

        return out

    def normalization(self, x):
        b, h, w = x.shape
        x = x.view(b, h * w)
        x_min = x.min(1, keepdim=True)[0]
        x_max = x.max(1, keepdim=True)[0]
        x = (x - x_min) / (x_max - x_min + 1e-7)
        x = x.view(b, 1, h, w)

        return x

    def _optimizer(self, args):
        optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, self.parameters()),
                                    lr=args.base_lr, 
                                    momentum=args.momentum, 
                                    weight_decay=args.weight_decay)
        return optimizer
