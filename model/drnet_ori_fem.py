import torch
import torch.nn.functional as F
from torch import nn

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
            param.requires_grad = False

        reduce_dim = 256
        if self.vgg:
            fea_dim = 512 + 256
        else:
            fea_dim = 1024 + 512

        self.cls = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(reduce_dim, classes, kernel_size=1)
        )

        # Encoder
        self.side3_1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, padding=0, bias=False),  # fc6
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.side4_1 = nn.Sequential(
            nn.Conv2d(1024 + 256, 256, kernel_size=1, padding=0, bias=False),  # fc6
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.side5_1 = nn.Sequential(
            nn.Conv2d(2048 + 256, 256, kernel_size=1, padding=0, bias=False),  # fc6
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.down_feat = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=1, padding=0, bias=False),  # fc6
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        # CR
        self.non_local = NonLocalBlock2D(reduce_dim)

        self.pyramid_bins = args.ppm_scales

        self.avgpool_list = nn.ModuleList()
        for bin in self.pyramid_bins:
            if bin > 1:
                self.avgpool_list.append(nn.AdaptiveAvgPool2d(bin))
        mask_add_num = 256
        self.init_merge = nn.ModuleList()
        self.beta_conv = nn.ModuleList()
        self.inner_cls = nn.ModuleList()
        for bin in self.pyramid_bins:
            self.init_merge.append(nn.Sequential(
                nn.Conv2d(reduce_dim * 2, reduce_dim, kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
            ))
            self.beta_conv.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True)
            ))
            self.inner_cls.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.1),
                nn.Conv2d(reduce_dim, classes, kernel_size=1)
            ))

        self.res1 = nn.Sequential(
            nn.Conv2d(reduce_dim * len(self.pyramid_bins), reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )
        self.res2 = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

        self.GAP = nn.AdaptiveAvgPool2d(1)

        self.alpha_conv = nn.ModuleList()
        for _ in range(len(self.pyramid_bins) - 1):
            self.alpha_conv.append(
                nn.Sequential(
                    nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.ReLU(inplace=True)
                ))

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
            side_out3_1 = self.side3_1(x2)
            side_out4_1 = self.side4_1(torch.cat((side_out3_1, x3), dim=1))
            side_out5_1 = self.side5_1(torch.cat((side_out4_1, x4), dim=1))
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

        sz = query_feat.shape[2]

        # Support Feature
        supp_feat_list = []
        supp_feat_neg_list = []
        mask_list = []
        mask_neg_list = []
        final_supp_list = []
        cr_list = []
        for i in range(self.shot):
            mask = (s_y[:, i, :, :] == 1).float().unsqueeze(1)
            mask_neg = (s_y[:, i, :, :] == 0).float().unsqueeze(1)
            mask_list.append(mask)
            mask_neg_list.append(mask_neg)

            supp_feat = self.extract_features(s_x[:, i, :, :, :], is_query=False)

            mask = F.interpolate(mask, size=(sz, sz), mode='bilinear', align_corners=True)
            mask_neg = F.interpolate(mask_neg, size=(sz, sz), mode='bilinear', align_corners=True)

            cr_out = self.non_local(query_feat, supp_feat * mask)
            cr_list.append(cr_out)

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
        rec_map = self.normalization(rec_map)

        out_list = []
        pyramid_feat_list = []

        for idx, tmp_bin in enumerate(self.pyramid_bins):
            if tmp_bin <= 1.0:
                bin = int(query_feat.shape[2] * tmp_bin)
                query_feat_bin = nn.AdaptiveAvgPool2d(bin)(query_feat_side)
            else:
                bin = tmp_bin
                query_feat_bin = self.avgpool_list[idx](query_feat_side)

            # supp_feat_bin = supp_feat.expand(-1, -1, bin, bin)
            cr_feat_bin = F.interpolate(cr_out, size=(bin, bin), mode='bilinear', align_corners=True)
            rec_map = F.interpolate(rec_map, size=(bin, bin), mode='bilinear', align_corners=True)
            sr_out_bin = query_feat_bin * rec_map
            # merge_feat_bin = torch.cat([query_feat_bin, supp_feat_bin, sr_out_bin, cr_feat_bin], 1)
            merge_feat_bin = torch.cat([sr_out_bin, cr_feat_bin], 1)
            merge_feat_bin = self.init_merge[idx](merge_feat_bin)

            if idx >= 1:
                pre_feat_bin = pyramid_feat_list[idx - 1].clone()
                pre_feat_bin = F.interpolate(pre_feat_bin, size=(bin, bin), mode='bilinear', align_corners=True)
                rec_feat_bin = torch.cat([merge_feat_bin, pre_feat_bin], 1)
                merge_feat_bin = self.alpha_conv[idx - 1](rec_feat_bin) + merge_feat_bin

            merge_feat_bin = self.beta_conv[idx](merge_feat_bin) + merge_feat_bin
            inner_out_bin = self.inner_cls[idx](merge_feat_bin)
            merge_feat_bin = F.interpolate(merge_feat_bin, size=(query_feat.size(2), query_feat.size(3)), mode='bilinear', align_corners=True)
            pyramid_feat_list.append(merge_feat_bin)
            out_list.append(inner_out_bin)

        query_feat = torch.cat(pyramid_feat_list, 1)
        query_feat = self.res1(query_feat)
        query_feat = self.res2(query_feat) + query_feat
        out = self.cls(query_feat)

        #   Output Part
        if self.zoom_factor != 1:
            out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)
            init_map = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)
            fin_out = out + 0.4 * init_map

        if self.training:
            main_loss = self.criterion(out, y.long())
            aux_loss = torch.zeros_like(main_loss)
            l_init = self.criterion(init_map, y.long())

            for idx_k in range(len(out_list)):
                inner_out = out_list[idx_k]
                inner_out = F.interpolate(inner_out, size=(h, w), mode='bilinear', align_corners=True)
                aux_loss = aux_loss + self.criterion(inner_out, y.long())
            aux_loss = aux_loss / len(out_list)

            return fin_out.max(1)[1], main_loss, aux_loss + 0.4 * l_init

        return fin_out

    def normalization(self, x):
        b, h, w = x.shape
        x = x.view(b, h * w)
        x_min = x.min(1, keepdim=True)[0]
        x_max = x.max(1, keepdim=True)[0]
        x = (x - x_min) / (x_max - x_min + 1e-7)
        x = x.view(b, 1, h, w)

        return x

    def aspp_pipeline(self, guide_feat):
        final_feat = self.corr_conv(guide_feat)
        final_feat = final_feat + self.skip1(final_feat)
        final_feat = final_feat + self.skip2(final_feat)
        final_feat = final_feat + self.skip3(final_feat)
        final_feat = self.ASPP(final_feat)
        out = self.cls(final_feat)

        return out

    def _optimizer(self, args):
        optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, self.parameters()),
                                    lr=args.base_lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        return optimizer
