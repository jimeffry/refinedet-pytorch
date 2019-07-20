#author: lxy
#time: 14:30 2019.7.2
#tool: python3
#version: 0.1
#project: pedestrian detection
#modify:
#####################################################
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import resnet101
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfgs
sys.path.append(os.path.join(os.path.dirname(__file__),'../utils'))
from prior_box import PriorBox
from l2norm import L2Norm


class RefineDet(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: RefineDet for more details.
    Args:
        size: input image size
        base: VGG16 layers for input, size of either 320 or 512
        extras: extra layers that feed to multibox loc and conf layers
        ARM: "default box head" consists of loc and conf conv layers
        ODM: "multibox head" consists of loc and conf conv layers
        TCB: converting the features from the ARM to the ODM for detection
        numclass:ODM output classes
    """
    def __init__(self, base, extras, ARM, ODM, TCB, num_classes):
        super(RefineDet, self).__init__()
        self.num_classes = num_classes
        self.priorbox = PriorBox(cfgs.PriorBox_Cfg[str(cfgs.ImgSize)])
        with torch.no_grad():
            self.priors = self.priorbox.forward()
        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.conv4_3_L2Norm = L2Norm(512, 10)
        self.conv5_3_L2Norm = L2Norm(512, 8)
        self.extras = nn.ModuleList(extras)
        self.arm_loc = nn.ModuleList(ARM[0])
        self.arm_conf = nn.ModuleList(ARM[1])
        self.odm_loc = nn.ModuleList(ODM[0])
        self.odm_conf = nn.ModuleList(ODM[1])
        #self.tcb = nn.ModuleList(TCB)
        self.tcb0 = nn.ModuleList(TCB[0])
        self.tcb1 = nn.ModuleList(TCB[1])
        self.tcb2 = nn.ModuleList(TCB[2])

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.
        Args:
            x: input image or batch of images. Shape: [batch,3,w,h].
        Return:
            list of concat outputs from:
                1: confidence layers, Shape: [batch*num_priors,num_classes]
                2: localization layers, Shape: [batch,num_priors*4]
                3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        tcb_source = list()
        arm_loc = list()
        arm_conf = list()
        odm_loc = list()
        odm_conf = list()
        odm_conf_debug = list()
        # apply vgg up to conv4_3 relu and conv5_3 relu
        for k in range(30):
            x = self.vgg[k](x)
            if 22 == k:
                s = self.conv4_3_L2Norm(x)
                sources.append(s)
            elif 29 == k:
                s = self.conv5_3_L2Norm(x)
                sources.append(s)
        # apply vgg up to fc7
        for k in range(30, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)
        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)
        # apply ARM and ODM to source layers
        for (x, l, c) in zip(sources, self.arm_loc, self.arm_conf):
            arm_loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            arm_conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        arm_loc = torch.cat([tmp.view(tmp.size(0), -1) for tmp in arm_loc], 1)
        arm_conf = torch.cat([tmp.view(tmp.size(0), -1) for tmp in arm_conf], 1)
        #print([x.size() for x in sources])
        # calculate TCB features
        #print([x.size() for x in sources])
        p = None
        for k, v in enumerate(sources[::-1]):
            s = v
            for i in range(3):
                s = self.tcb0[(3-k)*3 + i](s)
                #print(s.size())
            if k != 0:
                u = p
                u = self.tcb1[3-k](u)
                s += u
            for i in range(3):
                s = self.tcb2[(3-k)*3 + i](s)
            p = s
            tcb_source.append(s)
        #print([x.size() for x in tcb_source])
        tcb_source.reverse()
        # apply ODM to source layers
        for (x, l, c) in zip(tcb_source, self.odm_loc, self.odm_conf):
            odm_loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            odm_conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        odm_conf_debug = odm_conf
        odm_loc = torch.cat([tmp.view(tmp.size(0), -1) for tmp in odm_loc], 1)
        odm_conf = torch.cat([tmp.view(tmp.size(0), -1) for tmp in odm_conf], 1)
        #print(arm_loc.size(), arm_conf.size(), odm_loc.size(), odm_conf.size())
        output = (
                arm_loc.view(arm_loc.size(0), -1, 4),
                arm_conf.view(arm_conf.size(0), -1, 2),
                odm_loc.view(odm_loc.size(0), -1, 4),
                odm_conf.view(odm_conf.size(0), -1, self.num_classes),
                self.priors,
                odm_conf_debug
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        #device = torch.device('cpu')
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file),strict=False)
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=3, dilation=3)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, size, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers

def arm_multibox(vgg, extra_layers, cfg):
    arm_loc_layers = []
    arm_conf_layers = []
    vgg_source = [21, 28, -2]
    for k, v in enumerate(vgg_source):
        arm_loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        arm_conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * 2, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 3):
        arm_loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        arm_conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * 2, kernel_size=3, padding=1)]
    return (arm_loc_layers, arm_conf_layers)

def odm_multibox(vgg, extra_layers, cfg, num_classes):
    odm_loc_layers = []
    odm_conf_layers = []
    vgg_source = [21, 28, -2]
    for k, v in enumerate(vgg_source):
        odm_loc_layers += [nn.Conv2d(256, cfg[k] * 4, kernel_size=3, padding=1)]
        odm_conf_layers += [nn.Conv2d(256, cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 3):
        odm_loc_layers += [nn.Conv2d(256, cfg[k] * 4, kernel_size=3, padding=1)]
        odm_conf_layers += [nn.Conv2d(256, cfg[k] * num_classes, kernel_size=3, padding=1)]
    return (odm_loc_layers, odm_conf_layers)

def add_tcb(cfg):
    feature_scale_layers = []
    feature_upsample_layers = []
    feature_pred_layers = []
    for k, v in enumerate(cfg):
        feature_scale_layers += [nn.Conv2d(cfg[k], 256, 3, padding=1),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(256, 256, 3, padding=1)
        ]
        feature_pred_layers += [nn.ReLU(inplace=True),
                                nn.Conv2d(256, 256, 3, padding=1),
                                nn.ReLU(inplace=True)
        ]
        if k != len(cfg) - 1:
            feature_upsample_layers += [nn.ConvTranspose2d(256, 256, 2, 2)]#nn.Upsample(scale_factor=2, mode='bilinear')
    return (feature_scale_layers, feature_upsample_layers, feature_pred_layers)


def build_refinedet():
    size=cfgs.ImgSize
    num_classes = cfgs.ClsNum
    if size != 320 and size != 512:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only RefineDet320 and RefineDet512 is supported!")
        return
    base = cfgs.VGG_Base
    extras = cfgs.Extras
    mbox = cfgs.Scale_num
    tcb = cfgs.Tcb
    base_ = vgg(base[str(size)], 3)
    extras_ = add_extras(extras[str(size)], size, 1024)
    ARM_ = arm_multibox(base_, extras_, mbox[str(size)])
    ODM_ = odm_multibox(base_, extras_, mbox[str(size)], num_classes)
    TCB_ = add_tcb(tcb[str(size)])
    return RefineDet(base_, extras_, ARM_, ODM_, TCB_, num_classes)
