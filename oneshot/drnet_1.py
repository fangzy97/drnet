import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# from models.vgg import vgg_sg as vgg
from models.resnet import dres_feat as resnet

# from models.non_local_concatenation import NONLocalBlock2D

_BATCH_NORM = nn.BatchNorm2d
BatchNorm2d = nn.BatchNorm2d


class OCM(nn.Module):
    def __init__(self, category, in_features, delta=0.5):
        super(OCM, self).__init__()
        self.in_features = in_features
        self.category = category
        self.delta = delta
        self.weight = nn.Parameter(torch.empty(category, in_features), requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant(self.weight, 1 / math.sqrt(self.in_features))

    def forward(self, input, feat_dict):
        # forward
        output = torch.matmul(input, self.weight)
        output = output.permute(0, 2, 1)

        # backward
        for key, value in feat_dict.items():
            self.weight[key] = self.delta * self.weight[key] + (1 - self.delta) * value
            self.weight[key] = torch.norm(self.weight[key], 2)

        return output


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    "3x3 convolution + BN + relu"
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3,
                  stride=stride, padding=1, bias=False),
        BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )


class _ConvBnReLU(nn.Sequential):
    """
    Cascade of 2D convolution, batch norm, and ReLU.
    """

    BATCH_NORM = _BATCH_NORM

    def __init__(
            self, in_ch, out_ch, kernel_size, stride, padding, dilation, relu=True, dropout=False
    ):
        super(_ConvBnReLU, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_ch, out_ch, kernel_size, stride, padding, dilation, bias=False
            ),
        )
        # self.add_module("bn", _BATCH_NORM(out_ch, eps=1e-5, momentum=0.999))

        if relu:
            self.add_module("relu", nn.PReLU())

        if dropout:
            self.add_module("dropout", nn.Dropout2d(p=0.5))

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class _ImagePool(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(_ImagePool, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = _ConvBnReLU(in_ch, out_ch, 1, 1, 0, 1)

    def forward(self, x):
        _, _, H, W = x.shape
        h = self.pool(x)
        h = self.conv(h)
        h = F.interpolate(h, size=(H, W), mode="bilinear", align_corners=False)
        return h


class _ASPP(nn.Module):
    """
    Atrous spatial pyramid pooling with image-level feature
    """

    def __init__(self, in_ch, out_ch, rates):
        super(_ASPP, self).__init__()
        self.stages = nn.Module()
        for i, rate in enumerate(rates):
            if rate == 1:
                self.stages.add_module("c{}".format(i), _ConvBnReLU(in_ch, out_ch, 1, 1, padding=0, dilation=1))
            else:
                self.stages.add_module("c{}".format(i), _ConvBnReLU(in_ch, out_ch, 3, 1, padding=rate, dilation=rate))

        self.stages.add_module("imagepool", _ImagePool(in_ch, out_ch))

    def forward(self, x):
        return torch.cat([stage(x) for stage in self.stages.children()], dim=1)


class vanilla_residual_block(nn.Module):

    def __init__(self, inchannel):
        super(vanilla_residual_block, self).__init__()
        add_block = []
        add_block += [_ConvBnReLU(inchannel, 256, 3, 1, 1, 1, dropout=False)]
        add_block += [_ConvBnReLU(256, 256, 3, 1, 1, 1, dropout=False)]
        add_block = nn.Sequential(*add_block)
        self.add_block = add_block

    def forward(self, x, mode=1):
        if mode:
            x = self.add_block(x) + x
        else:
            x = self.add_block(x)
        return x


class IOM(nn.Module):
    def __init__(self):
        super(IOM, self).__init__()
        self._ASSP = _ASPP(256, 256, [1, 6, 12, 18])
        self.vanilla0 = vanilla_residual_block(256 + 256)
        self.vanilla1 = vanilla_residual_block(256)
        self.vanilla2 = vanilla_residual_block(256)
        self.fuse = _ConvBnReLU(256 * 5, 256, 1, 1, 0, 1, dropout=False)
        self.exit_layer = _ConvBnReLU(256, 2, 1, 1, 0, 1, dropout=False)

    def forward(self, x, i, mask):
        if i == 0:
            a = x
            x = self.vanilla1(x)
            x = self.vanilla2(x)
            x = self._ASSP(x)
            x = self.fuse(x)
            out = self.exit_layer(x)
            return a, out
        else:
            a = x
            x = torch.cat((x, mask), 1)
            x = self.vanilla0(x, mode=0)
            x = self.vanilla1(x)
            x = self.vanilla2(x)
            x = self._ASSP(x)
            x = self.fuse(x)
            out = self.exit_layer(x)

            return a, out


class OneModel(nn.Module):
    def __init__(self, args):
        super(OneModel, self).__init__()

        self.netB = resnet.drn_a_50(pretrained=True)
        self.IOM0 = IOM()
        self.bce_logits_func = nn.CrossEntropyLoss(ignore_index=255)
        self.loss_func = nn.BCELoss()
        self.cos_similarity_func = nn.CosineSimilarity()
        # self.combine = vanilla_residual_block(inchannel=2048)
        self.triplelet_func = nn.TripletMarginLoss(margin=2.0)
        self.context = _ConvBnReLU(2, 2, 1, 1, 0, 1)

    #         self.non_local3 = NONLocalBlock2D(in_channels=1024)
    def forward(self, anchor_img, pos_img, neg_img, pos_mask, category=0, group=None):
        outA = self.netB(pos_img)

        feat_mask = []

        h, w = outA.shape[-2:]
        mask = F.interpolate(pos_mask, size=(h, w), mode='nearest')
        A1 = outA * mask

        mask_w, mask_h = pos_mask.shape[-2:]
        outA_pos = F.interpolate(outA, size=(mask_w, mask_h), mode='bilinear', align_corners=True)

        vec_pos = torch.sum(torch.sum(outA_pos * pos_mask, dim=3), dim=2) / (
                    torch.sum(torch.sum(torch.sum(pos_mask, dim=3), dim=2), dim=1, keepdim=True) + 1e-5)
        vec_pos = vec_pos.unsqueeze(dim=2).unsqueeze(dim=3)

        feat_mask.append(A1)

        non, outB, outB_side = self.netB(anchor_img, feat_mask)
        tmp_seg = self.cos_similarity_func(outB, vec_pos)
        outB_side1 = outB_side * tmp_seg.unsqueeze(1)

        vec_neg = torch.sum(torch.sum(outA_pos * torch.abs(1 - pos_mask), dim=3), dim=2) / (torch.sum(
            torch.sum(torch.sum(torch.abs(1 - pos_mask), dim=3), dim=2), dim=1, keepdim=True) + 1e-5)

        vec_n = vec_neg.unsqueeze(dim=2).unsqueeze(dim=3)

        tmp_pos = self.cos_similarity_func(outB, vec_pos)
        tmp_pos = tmp_pos.unsqueeze(1)
        tmp_neg = self.cos_similarity_func(outB, vec_n)
        tmp_neg = tmp_neg.unsqueeze(1)

        # B*2*N
        output = torch.cat((tmp_neg, tmp_pos), dim=1) * 20
        pos1 = torch.max(output, dim=1, keepdim=True)[1]

        vec_pos1 = torch.sum(torch.sum(outB * pos1, dim=3), dim=2) / (torch.sum(
            torch.sum(torch.sum(pos1, dim=3), dim=2), dim=1, keepdim=True) + 1e-5)
        vec_pos1 = vec_pos1.unsqueeze(2).unsqueeze(dim=3)

        tmp_seg1 = self.cos_similarity_func(outB, vec_pos1)
        hehe = outB_side * tmp_seg1.unsqueeze(1)
        # B*C*2

        a, out = self.IOM0(outB_side1 + non, 1, hehe)
        return tmp_seg.unsqueeze(1), tmp_seg1.unsqueeze(1), output, out

    def forward_5shot_avg(self, anchor_img, pos_img_list, pos_mask_list):
        vec_pos_sum = 0.
        vec_neg_sum = 0.
        feat = []
        for i in range(5):
            pos_img = pos_img_list[i]
            pos_mask = pos_mask_list[i]

            pos_img = pos_img.unsqueeze(0)
            pos_mask = pos_mask.unsqueeze(0).unsqueeze(0)

            outA_pos = self.netB(pos_img)

            _, _, mask_w, mask_h = pos_mask.size()

            b, c, h, w = outA_pos.size()
            mask = F.interpolate(pos_mask, size=(h, w), mode='nearest')
            A1 = outA_pos * mask

            outA_pos = F.interpolate(outA_pos, size=(mask_w, mask_h), mode='bilinear', align_corners=True)
            vec_pos = torch.sum(torch.sum(outA_pos * pos_mask, dim=3), dim=2) / (torch.sum(
                torch.sum(torch.sum(pos_mask, dim=3), dim=2), dim=1, keepdim=True) + 1e-5)
            vec_pos = vec_pos.unsqueeze(dim=2).unsqueeze(dim=3)
            vec_neg = torch.sum(torch.sum(outA_pos * torch.abs(1 - pos_mask), dim=3), dim=2) / (torch.sum(
                torch.sum(torch.sum(torch.abs(1 - pos_mask), dim=3), dim=2), dim=1, keepdim=True) + 1e-5)
            vec_neg = vec_neg.unsqueeze(dim=2).unsqueeze(dim=3)
            vec_pos_sum += vec_pos
            vec_neg_sum += vec_neg
            feat.append(A1)

        vec_pos = vec_pos_sum / 5.0
        vec_neg = vec_neg_sum / 5.0

        non, outB, outB_side = self.netB(anchor_img, feat)

        tmp_pos = self.cos_similarity_func(outB, vec_pos)
        tmp_pos = tmp_pos.unsqueeze(1)
        tmp_neg = self.cos_similarity_func(outB, vec_neg)
        tmp_neg = tmp_neg.unsqueeze(1)

        # B*2*N
        output = torch.cat((tmp_neg, tmp_pos), dim=1) * 20
        pos1 = torch.max(output, dim=1, keepdim=True)[1]

        vec_pos1 = torch.sum(torch.sum(outB * pos1, dim=3), dim=2) / (torch.sum(
            torch.sum(torch.sum(pos1, dim=3), dim=2), dim=1, keepdim=True) + 1e-10)
        vec_pos1 = vec_pos1.unsqueeze(2).unsqueeze(dim=3)

        tmp_seg1 = self.cos_similarity_func(outB, vec_pos1)
        hehe = outB_side * tmp_seg1.unsqueeze(1)
        # tmp_seg = outB * vec_pos.unsqueeze(dim=2).unsqueeze(dim=3)
        # vec_pos = vec_pos.unsqueeze(dim=2).unsqueeze(dim=3)
        tmp_seg = self.cos_similarity_func(outB, vec_pos)

        exit_feat_in = outB_side * tmp_seg.unsqueeze(dim=1)
        # print(exit_feat_in.size())
        a, outB_side = self.IOM0(exit_feat_in + non, 1, hehe)

        return outB, outA_pos, output, outB_side

    def warper_img(self, img):
        img_tensor = torch.tensor(img).cuda()
        img_tensor.unsqueeze_(0)
        return img_tensor

    def forward_5shot_max(self, anchor_img, pos_img_list, pos_mask_list):
        outB_side_list = []
        for i in range(5):
            pos_img = pos_img_list[i]
            pos_mask = pos_mask_list[i]

            pos_img = self.warper_img(pos_img)
            pos_mask = self.warper_img(pos_mask)

            outA_pos, _ = self.netB(pos_img)

            _, _, mask_w, mask_h = pos_mask.size()
            outA_pos = F.interpolate(outA_pos, size=(mask_w, mask_h), mode='bilinear', align_corners=True)
            vec_pos = torch.sum(torch.sum(outA_pos * pos_mask, dim=3), dim=2) / torch.sum(pos_mask)

            outB, outB_side = self.netB(anchor_img)

            # tmp_seg = outB * vec_pos.unsqueeze(dim=2).unsqueeze(dim=3)
            vec_pos = vec_pos.unsqueeze(dim=2).unsqueeze(dim=3)
            tmp_seg = self.cos_similarity_func(outB, vec_pos)

            exit_feat_in = outB_side * tmp_seg.unsqueeze(dim=1)
            outB_side_6 = self.classifier_6(exit_feat_in)
            outB_side = self.exit_layer(outB_side_6)

            outB_side_list.append(outB_side)

        return outB, outA_pos, vec_pos, outB_side_list

    def get_loss(self, logits, query_label):
        # print(torch.max(query_label))
        outB, pos_mask, outA_side, outB_side = logits

        b, c, w, h = query_label.size()
        outB_side = F.interpolate(outB_side, size=(w, h), mode='bilinear', align_corners=True)
        query_label = query_label.squeeze(1)
        loss_bce_seg = self.bce_logits_func(outB_side, query_label.long())
        outA_side = F.interpolate(outA_side, size=(w, h), mode='bilinear', align_corners=True)
        loss_bce_seg2 = self.bce_logits_func(outA_side, query_label.long())
        loss = loss_bce_seg + loss_bce_seg2

        return 0, 0, loss, loss_bce_seg, loss

    def get_pred_5shot_max(self, logits, query_label):
        outB, outA_pos, vec_pos, outB_side_list = logits

        w, h = query_label.size()[-2:]
        res_pred = None
        for i in range(5):
            outB_side = outB_side_list[i]
            outB_side = F.interpolate(outB_side, size=(w, h), mode='bilinear', align_corners=True)
            out_side = F.softmax(outB_side, dim=1).squeeze()
            values, pred = torch.max(out_side, dim=0)

            if res_pred is None:
                res_pred = pred
            else:
                res_pred = torch.max(pred, res_pred)

        return values, res_pred

    def get_pred(self, logits, query_image):
        outB, outA_pos, vec_pos, outB_side = logits

        w, h = query_image.size()[-2:]
        outB_side = F.interpolate(outB_side, size=(w, h), mode='bilinear', align_corners=True)
        vec_pos = F.interpolate(vec_pos, size=(w, h), mode='bilinear', align_corners=True)

        out_softmax = F.softmax(vec_pos, dim=1).squeeze()
        coarse = out_softmax.argmax(dim=0)

        fuse = vec_pos + outB_side
        out_softmax = F.softmax(fuse, dim=1).squeeze()
        pred = out_softmax.argmax(dim=0)

        return fuse, pred, coarse
