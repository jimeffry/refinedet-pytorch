#author: lxy
#time: 14:30 2019.7.2
#tool: python3
#version: 0.1
#project: pedestrian detection
#modify:
#####################################################
import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import resnet101,resnet_layer5,FPN
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfgs
sys.path.append(os.path.join(os.path.dirname(__file__),'../utils'))
from prior_box import PriorBox
from l2norm import L2Norm

class RPN_Pred(nn.Module):
    def __init__(self,inplanes,planes,cls_num):
        super(RPN_Pred,self).__init__()
        self.conv1 = nn.Conv2d(inplanes, 512, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2_1 = nn.Conv2d(512,planes * 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(inplanes, 512, 3, padding=1)
        self.conv2_2 = nn.Conv2d(512,planes * cls_num, kernel_size=3, padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #print('layer5',m.weight.size())
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self,x):
        loc_x = self.conv1(x)
        relu1 = self.relu(loc_x)
        loc_x = self.conv2_1(relu1)
        conf_x = self.conv2(x)
        relu2 = self.relu(conf_x)
        conf_x = self.conv2_2(relu2)
        return loc_x,conf_x

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
    def __init__(self,fpn_filter_list,scale_list, num_classes):
        super(RefineDet, self).__init__()
        self.num_classes = num_classes
        self.priorbox = PriorBox(cfgs.PriorBox_Cfg_resnet[str(cfgs.ImgSize)])
        with torch.no_grad():
            self.priors = self.priorbox.forward()
        # SSD network
        inplanes = 2048
        planes = 512
        self.backone = resnet101(pretrained=True)
        self.res6 = resnet_layer5(inplanes, planes,3,2)
        self.FPN = FPN(fpn_filter_list)
        Arm_P3 = RPN_Pred(fpn_filter_list[0],scale_list[0],2)
        Arm_P4 = RPN_Pred(fpn_filter_list[1],scale_list[1],2)
        Arm_P5 = RPN_Pred(fpn_filter_list[2],scale_list[2],2)
        Arm_P6 = RPN_Pred(fpn_filter_list[3],scale_list[3],2)
        Odm_P3 = RPN_Pred(256,scale_list[0],num_classes)
        Odm_P4 = RPN_Pred(256,scale_list[1],num_classes)
        Odm_P5 = RPN_Pred(256,scale_list[2],num_classes)
        Odm_P6 = RPN_Pred(256,scale_list[3],num_classes)
        self.Arm_list = nn.ModuleList([Arm_P3,Arm_P4,Arm_P5,Arm_P6])
        self.Odm_list = nn.ModuleList([Odm_P3,Odm_P4,Odm_P5,Odm_P6])

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
        arm_loc = list()
        arm_conf = list()
        odm_loc = list()
        odm_conf = list()
        tcb_source = list()
        odm_conf_map = list()
        # apply vgg up to conv4_3 relu and conv5_3 relu
        c3,c4,c5,x = self.backone(x)
        c6 = self.res6(x)
        sources = [c3,c4,c5,c6]       
        # apply ARM  to source layers
        for (x, arm_pred) in zip(sources, self.Arm_list):
            arm_loc.append(arm_pred(x)[0].permute(0, 2, 3, 1).contiguous())
            arm_conf.append(arm_pred(x)[1].permute(0, 2, 3, 1).contiguous())      
        arm_loc = torch.cat([tmp.view(tmp.size(0), -1) for tmp in arm_loc], 1)
        arm_conf = torch.cat([tmp.view(tmp.size(0), -1) for tmp in arm_conf], 1)
        #apply tcb
        p3,p4,p5,p6 = self.FPN(c3,c4,c5,c6)
        tcb_source = [p3,p4,p5,p6]
        # apply ODM to source layers
        for (x, odm_pred) in zip(tcb_source, self.Odm_list):
            odm_loc.append(odm_pred(x)[0].permute(0, 2, 3, 1).contiguous())
            odm_conf.append(odm_pred(x)[1].permute(0, 2, 3, 1).contiguous())
        odm_conf_map = odm_conf
        odm_loc = torch.cat([tmp.view(tmp.size(0), -1) for tmp in odm_loc], 1)
        odm_conf = torch.cat([tmp.view(tmp.size(0), -1) for tmp in odm_conf], 1)
        #print(arm_loc.size(), arm_conf.size(), odm_loc.size(), odm_conf.size())
        output = (
                arm_loc.view(arm_loc.size(0), -1, 4),
                arm_conf.view(arm_conf.size(0), -1, 2),
                odm_loc.view(odm_loc.size(0), -1, 4),
                odm_conf.view(odm_conf.size(0), -1, self.num_classes),
                self.priors,
                odm_conf_map
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

def build_refinedet():
    size=cfgs.ImgSize
    num_classes = cfgs.ClsNum
    if size != 320 and size != 512 and size !=380:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only RefineDet320 and RefineDet512 is supported!")
        return
    mbox = cfgs.Scale_num
    feature_filters = cfgs.feature_map_filters
    return RefineDet(feature_filters,mbox[str(size)],num_classes)

if __name__=='__main__':
    net = build_refinedet()
    print(net)