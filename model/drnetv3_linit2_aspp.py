import torch
import torch.nn.functional as F
from torch import nn

from .lib.non_local import NonLocalBlock2D
from .pspnet import Model as PSPNet
from .lib.ASPP import ASPP


# Masked Average Pooling
def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat


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

        pspnet = PSPNet(args)
        backbone_str = 'vgg' if args.vgg else 'resnet' + str(args.layers)
        weight_path = 'initmodel/PSPNet/{}/split{}/{}/best.pth'.format(args.data_set, args.split, backbone_str)
        new_param = torch.load(weight_path, map_location=torch.device('cpu'))['state_dict']
        try: 
            pspnet.load_state_dict(new_param)
        except RuntimeError:                   # 1GPU loads mGPU model
            for key in list(new_param.keys()):
                new_param[key[7:]] = new_param.pop(key)
            pspnet.load_state_dict(new_param)

        self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = pspnet.layer0, pspnet.layer1, pspnet.layer2, pspnet.layer3, pspnet.layer4

        for param in self.parameters():
            param.requires_grad = False;

        reduce_dim = 256
        if self.vgg:
            fea_dim = 512 + 256
        else:
            fea_dim = 1024 + 512 

        # Encoder
        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.down_supp = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        # CR
        self.non_local = NonLocalBlock2D(reduce_dim)

        mask_add_num = 0
        self.init_merge = nn.Sequential(
            nn.Conv2d(reduce_dim * 2 + mask_add_num, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True))
        self.ASPP = ASPP(reduce_dim)
        self.res1 = nn.Sequential(
            nn.Conv2d(reduce_dim*5, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True))
        self.res2 = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),   
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True))
        self.cls = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(reduce_dim, args.classes, kernel_size=1))
        
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

    def extract_features(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x_2 = self.layer2(x)
        x_3 = self.layer3(x_2)
        x_4 = self.layer4(x_3)

        return x_2, x_3, x_4

    def forward(self, x, s_x, s_y, y=None, cat_idx=None):
        self.freeze_backbone_bn()

        x_size = x.size()
        assert (x_size[2] - 1) % 8 == 0 and (x_size[3] - 1) % 8 == 0
        h = int((x_size[2] - 1) // 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) // 8 * self.zoom_factor + 1)

        # Query Feature
        with torch.no_grad():
            query_feat_2, query_feat_3, query_feat_4 = self.extract_features(x)
            if self.vgg:
                query_feat_2 = F.interpolate(query_feat_2, size=(query_feat_3.size(2),query_feat_3.size(3)), mode='bilinear', align_corners=True)
        query_feat = torch.cat([query_feat_3, query_feat_2], 1)
        query_feat = self.down_query(query_feat)

        sz = query_feat.shape[2]

        # Support Feature
        supp_feat_list = []
        mask_list = []
        mask_neg_list = []
        final_supp_list = []
        cr_list = []
        for i in range(self.shot):
            mask = (s_y[:, i, :, :] == 1).float().unsqueeze(1)
            mask_neg = (s_y[:, i, :, :] == 0).float().unsqueeze(1)
            mask_list.append(mask)
            mask_neg_list.append(mask_neg)

            with torch.no_grad():
                supp_feat_2, supp_feat_3, supp_feat_4 = self.extract_features(s_x[:, i, :, :, :])
                if self.vgg:
                    supp_feat_2 = F.interpolate(supp_feat_2, size=(supp_feat_3.size(2),supp_feat_3.size(3)), mode='bilinear', align_corners=True)
            supp_feat = torch.cat([supp_feat_3, supp_feat_2], 1)
            supp_feat = self.down_supp(supp_feat)

            mask = F.interpolate(mask, size=(sz, sz), mode='bilinear', align_corners=True)
            cr_out = self.non_local(query_feat, supp_feat * mask)
            cr_list.append(cr_out)

            supp_feat = Weighted_GAP(supp_feat, mask)
            supp_feat_list.append(supp_feat)

            final_supp_list.append(supp_feat_4)

        supp_feat = sum(supp_feat_list) / len(supp_feat_list)
        cr_out = sum(cr_list) / len(cr_list)

        # SR module
        sr_out_list = []
        init_map_list = []
        for i, tmp_supp_feat in enumerate(final_supp_list):
            resize_size = tmp_supp_feat.size(2)
            tmp_mask = F.interpolate(mask_list[i], size=(resize_size, resize_size), mode='bilinear', align_corners=True)
            tmp_mask_neg = F.interpolate(mask_neg_list[i], size=(resize_size, resize_size), mode='bilinear', align_corners=True)

            fg_vectors = Weighted_GAP(tmp_supp_feat, tmp_mask)
            bg_vectors = Weighted_GAP(tmp_supp_feat, tmp_mask_neg)
            fg_init_map = torch.cosine_similarity(query_feat_4, fg_vectors)
            bg_init_map = torch.cosine_similarity(query_feat_4, bg_vectors)
            init_map = torch.stack((bg_init_map, fg_init_map), dim=1)
            init_map_list.append(init_map)

            init_mask = init_map.argmax(dim=1, keepdim=True)
            rec_vector = Weighted_GAP(query_feat_4, init_mask.float())
            rec_map = torch.cosine_similarity(query_feat_4, rec_vector)
            rec_map = self.normalization(rec_map)

            rec_map = F.interpolate(rec_map, size=(sz, sz), mode='bilinear', align_corners=True)
            sr_out_list.append(rec_map)
        sr_out = sum(sr_out_list) / len(sr_out_list)
        sr_out = F.interpolate(sr_out, size=(sz, sz), mode='bilinear', align_corners=True)

        init_map = sum(init_map_list) / len(init_map_list)

        merge_feat = torch.cat([cr_out, sr_out * query_feat], 1)   # 256 + 256 + 1
        merge_feat = self.init_merge(merge_feat)

        query_feat = self.ASPP(merge_feat)
        query_feat = self.res1(query_feat)
        query_feat = self.res2(query_feat) + query_feat
        out = self.cls(query_feat)

        #   Output Part
        if self.zoom_factor != 1:
            out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)
            init_map = F.interpolate(init_map, size=(h, w), mode='bilinear', align_corners=True)
        
        fin_out = out + 0.4 * init_map

        if self.training:
            main_loss = self.criterion(out, y.long())
            l_init = self.criterion(init_map, y.long())

            return fin_out.max(1)[1], main_loss, 0.4 * l_init

        sr_out = F.interpolate(sr_out, size=(h, w), mode='bilinear', align_corners=True)
        return fin_out, init_map, sr_out

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
