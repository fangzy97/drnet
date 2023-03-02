from pip import main
import torch
import torch.nn.functional as F
from torch import nn

from .lib.transformer.cyc_transformer import CyCTransformer
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
        self.with_transformer = True
        self.trans_multi_lvl = 1

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
            param.requires_grad = False

        reduce_dim = 256
        fea_dim = 1024 + 512

        self.cls = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(reduce_dim, classes, kernel_size=1)
        )

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
        mask_add_num = 1
        self.init_merge = nn.Sequential(
            nn.Conv2d(reduce_dim * 2 + mask_add_num, reduce_dim, kernel_size=1, padding=0, bias=False)
        )
        if self.with_transformer:
            self.cr = CyCTransformer(embed_dims=reduce_dim, shot=self.shot, num_points=9, num_heads=1, use_self=True)
            self.supp_merge_feat = nn.Sequential(
                nn.Conv2d(reduce_dim * 2, reduce_dim,
                          kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
            )
            self.merge_multi_lvl_reduce = nn.Sequential(
                nn.Conv2d(reduce_dim * self.trans_multi_lvl, reduce_dim,
                          kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=True)
            )
            self.merge_multi_lvl_sum = nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim,
                          kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduce_dim, reduce_dim,
                          kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
            )
        else:
            self.merge_res = nn.Sequential(
                    nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                    nn.ReLU(inplace=True),
                )
        self.ASPP = ASPP(reduce_dim)
        self.res1 = nn.Sequential(
            nn.Conv2d(reduce_dim*5, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True))
        self.res2 = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),   
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True))

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

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

    def forward(self, x, s_x, s_y, y, padding_mask=None, s_padding_mask=None):
        self.freeze_backbone_bn()

        x_size = x.size()
        assert (x_size[2] - 1) % 8 == 0 and (x_size[3] - 1) % 8 == 0
        h = int((x_size[2] - 1) // 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) // 8 * self.zoom_factor + 1)

        # Query Feature
        with torch.no_grad():
            query_feat_2, query_feat_3, query_feat_4 = self.extract_features(x)
        query_feat = torch.cat([query_feat_3, query_feat_2], 1)
        query_feat = self.down_query(query_feat)

        sz = query_feat.shape[2:]

        # Support Feature
        supp_feat_list = []  # support prototypes
        supp_feat_mid_list = []  # mid-level support features
        mask_list = []
        mask_neg_list = []
        final_supp_list = []
        for i in range(self.shot):
            mask = (s_y[:, i, :, :] == 1).float().unsqueeze(1)
            mask_neg = (s_y[:, i, :, :] == 0).float().unsqueeze(1)
            mask_list.append(mask)
            mask_neg_list.append(mask_neg)

            with torch.no_grad():
                supp_feat_2, supp_feat_3, supp_feat_4 = self.extract_features(s_x[:, i, :, :, :])
            supp_feat_mid = torch.cat([supp_feat_3, supp_feat_2], 1)
            supp_feat_mid = self.down_supp(supp_feat_mid)
            supp_feat_mid_list.append(supp_feat_mid)

            mask = F.interpolate(mask, size=(
                sz[0], sz[1]), mode='bilinear', align_corners=True)
            supp_feat = Weighted_GAP(supp_feat_mid, mask)
            supp_feat_list.append(supp_feat)

            final_supp_list.append(supp_feat_4)

        # SR module
        sr_out_list = []
        for i, tmp_supp_feat in enumerate(final_supp_list):
            resize_size = tmp_supp_feat.size(2)
            tmp_mask = F.interpolate(mask_list[i], size=(
                resize_size, resize_size), mode='bilinear', align_corners=True)
            tmp_mask_neg = F.interpolate(mask_neg_list[i], size=(
                resize_size, resize_size), mode='bilinear', align_corners=True)

            fg_vectors = Weighted_GAP(tmp_supp_feat, tmp_mask)
            bg_vectors = Weighted_GAP(tmp_supp_feat, tmp_mask_neg)
            fg_init_map = torch.cosine_similarity(query_feat_4, fg_vectors)
            bg_init_map = torch.cosine_similarity(query_feat_4, bg_vectors)
            init_map = torch.stack((bg_init_map, fg_init_map), dim=1)
            init_mask = init_map.argmax(dim=1, keepdim=True)
            rec_vector = Weighted_GAP(query_feat_4, init_mask.float())
            rec_map = torch.cosine_similarity(query_feat_4, rec_vector)
            rec_map = self.normalization(rec_map)

            rec_map = F.interpolate(rec_map, size=(sz[0], sz[1]), mode='bilinear', align_corners=True)
            sr_out_list.append(rec_map)
        sr_out = sum(sr_out_list) / len(sr_out_list)
        sr_out = F.interpolate(sr_out, size=(sz[0], sz[1]), mode='bilinear', align_corners=True)
        del tmp_mask, tmp_mask_neg, fg_vectors, bg_vectors, fg_init_map, bg_init_map, init_map
        del init_mask, rec_vector, rec_map

        # [bs x shot] x c x h x w
        supp_feat_mid = torch.cat(supp_feat_mid_list, dim=0)
        supp_feat = sum(supp_feat_list) / len(supp_feat_list)
        del supp_feat_mid_list, supp_feat_list, mask, mask_neg, supp_feat_2, supp_feat_3, supp_feat_4

        query_feat_cat = torch.cat([query_feat, supp_feat.expand_as(query_feat), sr_out], dim=1)
        query_feat = self.init_merge(query_feat_cat)

        if self.with_transformer:
            to_merge_fts = [supp_feat_mid, supp_feat.expand(-1, -1, sz[0], sz[1])]
            aug_supp_feat = torch.cat(to_merge_fts, dim=1)
            aug_supp_feat = self.supp_merge_feat(aug_supp_feat)

            query_feat_list = self.cr(query_feat, padding_mask.float(), aug_supp_feat, s_y.clone().float(), s_padding_mask.float())
            fused_query_feat = []
            for lvl, cr in enumerate(query_feat_list):
                if lvl == 0:
                    fused_query_feat.append(cr)
                else:
                    fused_query_feat.append(F.interpolate(cr, size=(sz[0], sz[1]), mode='bilinear', align_corners=True))
            fused_query_feat = torch.cat(fused_query_feat, dim=1)
            fused_query_feat = self.merge_multi_lvl_reduce(fused_query_feat)
            fused_query_feat = self.merge_multi_lvl_sum(fused_query_feat) + fused_query_feat
            del to_merge_fts, aug_supp_feat
        else:
            query_feat = self.merge_res(query_feat) + query_feat
            query_feat_list = [query_feat]
            fused_query_feat = query_feat.clone()

        query_feat = self.ASPP(fused_query_feat)
        query_feat = self.res1(query_feat)
        query_feat = self.res2(query_feat) + query_feat
        out = self.cls(query_feat)
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)

        if self.training:
            main_loss = self.criterion(out, y.long())         
            aux_loss = torch.zeros_like(main_loss)

            return out.max(1)[1], main_loss, aux_loss

        return out

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
        param_dicts = [{"params": [
            p for n, p in self.named_parameters() if "cr" not in n and p.requires_grad]}]
        transformer_param_dicts = [
            {
                "params": [p for n, p in self.named_parameters() if "cr" in n and "bias" not in n and p.requires_grad],
                "lr": 1e-4,
                "weight_decay": 1e-2,
            },
            {
                "params": [p for n, p in self.named_parameters() if "cr" in n and "bias" in n and p.requires_grad],
                "lr": 1e-4,
                "weight_decay": 0,
            }
        ]
        optimizer = torch.optim.SGD(param_dicts,
                                    lr=args.base_lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

        optimizer_transformer = torch.optim.AdamW(transformer_param_dicts,
                                                  lr=1e-4,
                                                  weight_decay=1e-4)
        return optimizer, optimizer_transformer
