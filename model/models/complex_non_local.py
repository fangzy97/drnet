import torch
from torch import nn
from torch.nn import functional as F

# from models.tools.module_helper import ModuleHelper


class PSPModule(nn.Module):
    # (1, 2, 3, 6)
    def __init__(self, sizes=(6, 8, 10), dimension=2):
        super(PSPModule, self).__init__()
        self.stages = nn.ModuleList([self._make_stage(size, dimension) for size in sizes])

    def _make_stage(self, size, dimension=2):
        if dimension == 1:
            prior = nn.AdaptiveAvgPool1d(output_size=size)
        elif dimension == 2:
            prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        elif dimension == 3:
            prior = nn.AdaptiveAvgPool3d(output_size=(size, size, size))
        return prior

    def forward(self, feats, mask):
        n, c, _, _ = feats.size()
        # priors = [(stage(feats)/stage(mask).clamp(1e-10)).view(n, c, -1) for stage in self.stages]
        priors = [stage(feats).view(n, c, -1) for stage in self.stages]

        center = torch.cat(priors, -1)
        return center


class _SelfAttentionBlock(nn.Module):
    '''
    The basic implementation for self-attention block/non-local block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        value_channels    : the dimension after the value transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
    Return:
        N X C X H X W
        position-aware context features.(w/o concate or add with the input)
    '''

    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1, norm_type=None,psp_size=(20)):
        super(_SelfAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        if out_channels == None:
            self.out_channels = in_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))

        self.dimension = 2
        self.sub_sample = True

        self.in_channels = in_channels
        self.inter_channels = None

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if self.dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif self.dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(
            conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
            bn(self.in_channels)
        )
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        # if self.sub_sample:
        #     self.g = nn.Sequential(self.g, max_pool_layer)
        #     self.phi = nn.Sequential(self.phi, max_pool_layer)

        self.psp = PSPModule(psp_size)
        # nn.init.constant_(self.W.weight, 0)
        # nn.init.constant_(self.W.bias, 0)

    def forward(self, x, y, mask):
        # batch_size, h, w = x.size(0), x.size(2), x.size(3)
        # if self.scale > 1:
        #     x = self.pool(x)
        #
        # # value = self.psp(self.f_value(y))
        # value = self.f_value(y).view(batch_size, self.key_channels, -1)
        # # print(value.size())
        # query = self.f_query(x).view(batch_size, self.key_channels, -1)
        # query = query.permute(0, 2, 1)
        # key = self.f_key(y).view(batch_size, self.key_channels, -1)
        # # value=self.psp(value)#.view(batch_size, self.value_channels, -1)
        # value = value.permute(0, 2, 1)
        # #key = self.psp(key)  # .view(batch_size, self.key_channels, -1)
        # sim_map = torch.matmul(query, key)
        # N = sim_map.size(-1)
        # sim_map = sim_map / N
        # # sim_map = (self.key_channels ** -.5) * sim_map
        # # sim_map = F.softmax(sim_map, dim=-1)
        #
        # context = torch.matmul(sim_map, value)
        # context = context.permute(0, 2, 1).contiguous()
        # context = context.view(batch_size, self.value_channels, *x.size()[2:])
        # context = self.W(context)

        batch_size = x.size(0)

        g_x = self.g(x)
        g_x = self.psp(g_x, mask)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(y).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x)
        phi_x = self.psp(phi_x, mask)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        return W_y


class SelfAttentionBlock2D(_SelfAttentionBlock):
    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1, norm_type=None,psp_size=(20)):
        super(SelfAttentionBlock2D, self).__init__(in_channels,
                                                   key_channels,
                                                   value_channels,
                                                   out_channels,
                                                   scale,
                                                   norm_type,
                                                   psp_size=psp_size)


class APNB(nn.Module):
    """
    Parameters:
        in_features / out_features: the channels of the input / output feature maps.
        dropout: we choose 0.05 as the default value.
        size: you can apply multiple sizes. Here we only use one size.
    Return:
        features fused with Object context information.
    """

    def __init__(self, in_channels, out_channels, key_channels, value_channels, dropout, sizes=([1]), norm_type=None,psp_size=(2, 10, 40)):
        super(APNB, self).__init__()
        self.stages = []
        self.norm_type = norm_type
        self.psp_size=psp_size
        self.stages = nn.ModuleList(
            [self._make_stage(in_channels, out_channels, key_channels, value_channels, size) for size in sizes])
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(2 * in_channels, out_channels, kernel_size=1, padding=0),
            # ModuleHelper.BNReLU(out_channels, norm_type=norm_type),
            nn.Dropout2d(dropout)
        )

    def _make_stage(self, in_channels, output_channels, key_channels, value_channels, size):
        return SelfAttentionBlock2D(in_channels,
                                    key_channels,
                                    value_channels,
                                    output_channels,
                                    size,
                                    self.norm_type,
                                    self.psp_size)

    def forward(self, feats, y, mask):
        priors = [stage(feats, y, mask) for stage in self.stages]
        context = priors[0]
        # print(len(priors))
        for i in range(1, len(priors)):
            context += priors[i]
        # output = self.conv_bn_dropout(torch.cat([context, feats], 1))

        return feats + context