import torch
from torch import nn
import torch.nn.functional as F

from model.lib.ASPP import ASPP
from model.pspnet import Model as PSPNet
from model.lib.mask2former_decoder import MultiScaleMaskedTransformerDecoder
from model.lib.mask2former_criterion import SetCriterion
from model.lib.mask2former_matcher import HungarianMatcher


def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat

def get_gram_matrix(fea):
    b, c, h, w = fea.shape        
    fea = fea.reshape(b, c, h*w)    # C*N
    fea_T = fea.permute(0, 2, 1)    # N*C
    fea_norm = fea.norm(2, 2, True)
    fea_T_norm = fea_T.norm(2, 1, True)
    gram = torch.bmm(fea, fea_T)/(torch.bmm(fea_norm, fea_T_norm) + 1e-7)    # C*C
    return gram


class Model(nn.Module):
    def __init__(self, args, cls_type='Base'):
        super(Model, self).__init__()

        self.cls_type = cls_type  # 'Base' or 'Novel'
        self.layers = args.layers
        self.zoom_factor = args.zoom_factor
        self.shot = args.shot
        self.vgg = args.vgg
        self.dataset = args.data_set
        self.criterion = SetCriterion(2, HungarianMatcher(), weight_dict={"loss_ce": 2.0, "loss_mask": 5.0, "loss_dice": 5.0},
                                      eos_coef=0.1, losses=['labels', 'masks'], num_points=0., oversample_ratio=0., importance_sample_ratio=0.)

        self.print_freq = args.print_freq/2

        self.pretrained = True
        self.classes = 2
        if self.dataset == 'pascal':
            self.base_classes = 15
        elif self.dataset == 'coco':
            self.base_classes = 60
        
        assert self.layers in [50, 101, 152]
    
        PSPNet_ = PSPNet(args)
        backbone_str = 'vgg' if args.vgg else 'resnet'+str(args.layers)
        weight_path = 'initmodel/PSPNet/{}/split{}/{}/best.pth'.format(args.data_set, args.split, backbone_str)               
        new_param = torch.load(weight_path, map_location=torch.device('cpu'))['state_dict']
        try: 
            PSPNet_.load_state_dict(new_param)
        except RuntimeError:                   # 1GPU loads mGPU model
            for key in list(new_param.keys()):
                new_param[key[7:]] = new_param.pop(key)
            PSPNet_.load_state_dict(new_param)

        self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = PSPNet_.layer0, PSPNet_.layer1, PSPNet_.layer2, PSPNet_.layer3, PSPNet_.layer4

        # Base Learner
        self.learner_base = nn.Sequential(PSPNet_.ppm, PSPNet_.cls)
        for param in self.parameters():
            param.requires_grad = False

        self.transformer_decoder = MultiScaleMaskedTransformerDecoder(256)

        # Meta Learner
        reduce_dim = [256, 512, 1024, 2048]

        mask_add_num = 1
        self.init_merge = nn.ModuleList()
        for dim in reduce_dim:
            self.init_merge.append(nn.Sequential(
                nn.Conv2d(dim * 2 + 1, 256, kernel_size=1),
                nn.ReLU(inplace=True)
            ))

    def _optimizer(self, args):
        sgd_parameters = [p for n, p in self.named_parameters() if p.requires_grad and 'transformer' not in n]
        optimizer = torch.optim.SGD(sgd_parameters,
                                    lr = args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
        adamw_parameters = [p for n, p in self.named_parameters() if p.requires_grad and 'transformer' in n]
        optimizer_trans = torch.optim.AdamW(adamw_parameters,
                                            lr=1e-4, weight_decay=1e-4)
        return optimizer, optimizer_trans


    # que_img, sup_img, sup_mask, que_mask(meta), que_mask(base), cat_idx(meta)
    def forward(self, x, s_x, s_y, y, padding_mask=None, s_padding_mask=None):
        x_size = x.size()
        bs = x_size[0]
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        # Query Feature
        with torch.no_grad():
            query_feat_0 = self.layer0(x)
            query_feat_1 = self.layer1(query_feat_0)
            query_feat_2 = self.layer2(query_feat_1)
            query_feat_3 = self.layer3(query_feat_2)
            query_feat_4 = self.layer4(query_feat_3)
        query_feat_list = [query_feat_1, query_feat_2, query_feat_3, query_feat_4]

        # Support Feature
        mask_list = []
        merged_feat_list = []
        for i in range(self.shot):
            mask = (s_y[:, i, :, :] == 1).float().unsqueeze(1)
            mask_list.append(mask)
            with torch.no_grad():
                supp_feat_0 = self.layer0(s_x[:, i, :, :, :])
                supp_feat_1 = self.layer1(supp_feat_0)
                supp_feat_2 = self.layer2(supp_feat_1)
                supp_feat_3 = self.layer3(supp_feat_2)
                supp_feat_4 = self.layer4(supp_feat_3)
            supp_feat_list = [supp_feat_1, supp_feat_2, supp_feat_3, supp_feat_4]
            merged_feats = [self.get_prior_mask(s, mask, q, merge)
                            for s, q, merge in zip(supp_feat_list, query_feat_list, self.init_merge)]
            merged_feat_list.append(merged_feats)
        # merged_feat = sum(merged_feat_list) / len(merged_feat_list)  # b x lvl x c x h x w
        # merged_feat = [merged_feat[:, i, :, :, :] for i in range(merged_feat.shape[1])]
        merged_feat = merged_feat_list[0]
        outputs = self.transformer_decoder(merged_feat[:3], merged_feat[3])
        targets = self.prepare_targets(y)
        losses = self.criterion(outputs, targets)

        pred_masks = outputs['pred_masks']
        pred_logits = outputs['pred_logits']
        out = torch.einsum('bqc,bqhw->bchw', pred_logits, pred_masks)
        if self.training:
            return out.argmax(dim=1), losses['labels'], losses['masks']
        return out

    def prepare_targets(self, targets):
        new_targets = []
        targets = F.interpolate(targets[:, None, :, :].float(), (60, 60), mode='nearest')
        targets = targets.squeeze(1).long()
        for targets_per_image in targets:
            # pad gt
            t_target = targets_per_image.unsqueeze(0).repeat(100, 1, 1)
            new_targets.append(
                {
                    "labels": 1,
                    "masks": t_target,
                }
            )
        return new_targets

    def get_prior_mask(self, supp_feat, supp_mask, query_feat, merge, cosine_eps=1e-7):
        resize_size = supp_feat.size(2)
        tmp_mask = F.interpolate(supp_mask, size=(resize_size, resize_size), mode='bilinear', align_corners=True)

        supp_pro = Weighted_GAP(supp_feat, tmp_mask)

        bsize, ch_sz, sp_sz, _ = query_feat.size()[:]
        tmp_query = query_feat.reshape(bsize, ch_sz, -1)
        tmp_query_norm = torch.norm(tmp_query, 2, 1, True)
            
        tmp_supp = (supp_feat * tmp_mask).reshape(bsize, ch_sz, -1) 
        tmp_supp = tmp_supp.permute(0, 2, 1)
        tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)

        similarity = torch.bmm(tmp_supp, tmp_query)/(torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)   
        similarity = similarity.max(1)[0].reshape(bsize, sp_sz*sp_sz)
        similarity = (similarity - similarity.min(1)[0].unsqueeze(1))/(similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
        corr_query = similarity.reshape(bsize, 1, sp_sz, sp_sz)

        concat_feat = supp_pro.expand_as(query_feat)
        merge_feat = torch.cat([query_feat, concat_feat, corr_query], 1)   # 256+256+1
        merge_feat = merge(merge_feat)

        return merge_feat