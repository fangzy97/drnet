import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import torch
import torchvision
from models.non_local1 import NONLocalBlock2D
BatchNorm = nn.BatchNorm2d

# __all__ = ['DRN', 'drn26', 'drn42', 'drn58']


webroot = 'http://dl.yf.io/drn/'

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'drn-c-26': webroot + 'drn_c_26-ddedf421.pth',
    'drn-c-42': webroot + 'drn_c_42-9d336e8c.pth',
    'drn-c-58': webroot + 'drn_c_58-0a53a92c.pth',
    'drn-d-22': webroot + 'drn_d_22-4bd2f8ea.pth',
    'drn-d-38': webroot + 'drn_d_38-eebb45f0.pth',
    'drn-d-54': webroot + 'drn_d_54-0e0534ff.pth',
    'drn-d-105': webroot + 'drn_d_105-12b40979.pth'
}


def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 dilation=(1, 1), residual=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride,
                             padding=dilation[0], dilation=dilation[0])
        self.bn1 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes,
                             padding=dilation[1], dilation=dilation[1])
        self.bn2 = BatchNorm(planes)
        self.downsample = downsample
        self.stride = stride
        self.residual = residual

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        if self.residual:
            out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 dilation=(1, 1), residual=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation[1], bias=False,
                               dilation=dilation[1])
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self, x, y):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = y.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

class DRN_A(nn.Module):

    def __init__(self, block, layers, side=False):
        self.side = side
        self.inplanes = 64
        super(DRN_A, self).__init__()
        self.decoder = True
        self.out_dim = 512 * block.expansion
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                       dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                       dilation=4)

        self.non_local1 = NONLocalBlock2D(in_channels=256)
        self.non_local2 = NONLocalBlock2D(in_channels=256)
        self.non_local3 = NONLocalBlock2D(in_channels=256)
        self.cam = CAM_Module(in_dim=256)

        self.cls = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=3, padding=1),  # fc6
        )

        self.com = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),  # fc6
        )
        # self.cam = CAM_Module(256)

        if self.decoder:
            self.side3_1 =  nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=3, dilation=1,  padding=1),   #fc6
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )
            self.side4_1 =  nn.Sequential(
                nn.Conv2d(1024+256, 256, kernel_size=3, dilation=1,  padding=1),   #fc6
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            )
            self.side5_1 =  nn.Sequential(
                nn.Conv2d(2048+256, 256, kernel_size=3, dilation=1,  padding=1),   #fc6
            )

        # self.cls2 = nn.Sequential(
        #     nn.Conv2d(2048, 512, kernel_size=3, padding=1),  # fc6
        #     nn.PReLU(),
        #     nn.BatchNorm2d(512),
        # )

        #  self.avgpool = nn.AvgPool2d(28, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight.data)
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                dilation=(dilation, dilation)))

        return nn.Sequential(*layers)

    def forward(self, x, feat=[]):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x0 = self.maxpool(x)
        if len(feat) == 0:
            x1 = self.layer1(x0)
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)
            x4 = self.layer4(x3)
            out = self.cls(x4)

            return out
        else:
            # assert len(feat) == 3
            x1 = self.layer1(x0)
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)
            x4 = self.layer4(x3)
            hehe = self.cls(x4)

            if self.decoder:
                side_out3_1 = self.side3_1(x2)
                side_out4_1 = self.side4_1(torch.cat((side_out3_1, x3), dim=1))
                side_out5_1 = self.side5_1(torch.cat((side_out4_1, x4), dim=1))


            non = 0.
            for i in range(len(feat)):
                non1 = self.non_local1(hehe, feat[i])
                non += non1
            # non2 = self.non_local2(out2, feat[1])
            # non3 = self.non_local3(hehe, feat[2])

            # non_fore = self.cam(non_fore, feat[0])
            # non_back = self.non_local2(hehe, feat[1], hehe)
            # non3 = self.non_local3(side_out5_1, feat[2])
            # non_back = self.non_local2(hehe, feat[1], side_out5_1)
            # non = self.combine(torch.cat((non_fore, non_back), dim=1))
            # non = non_fore - non_back + hehe
            out = non/len(feat)
            # out = self.cam(out, feat[0])
            return out, hehe, side_out5_1



def load_resnet50_param(model, stop_layer='fc'):
    resnet50 = torchvision.models.resnet50(pretrained=True)
    saved_state_dict = resnet50.state_dict()
    new_params = model.state_dict().copy()

    for i in saved_state_dict:  # copy params from resnet50,except layers after stop_layer

        i_parts = i.split('.')
        if not i_parts[0] == stop_layer:

            new_params['.'.join(i_parts)] = saved_state_dict[i]
        elif i_parts[0] == 'fc':
            break
        else:
            break
    model.load_state_dict(new_params)
    model.train()
    return model


def drn_a_50(pretrained=False, **kwargs):
    model = DRN_A(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model = load_resnet50_param(model)
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        # restore(model, model_zoo.load_url(model_urls['resnet50']))
    return model