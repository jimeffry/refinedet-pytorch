# config.py
import os
import sys
from easydict import EasyDict
cfgs = EasyDict()

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))
#************************************************************dataset
cfgs.ClsNum = 9
cfgs.COCODataNames = ['person','bicycle','motorcycle','car','bus','airplane','train','boat']
cfgs.VOCDataNames = ['person','bicycle','motorbike','car','bus','aeroplane','train','boat']
cfgs.PIXEL_MEAN = [0.485,0.456,0.406] # R, G, B
cfgs.PIXEL_NORM = [0.229,0.224,0.225] #rgb
cfgs.variance = [0.1, 0.2]
#**********************************************************************train
cfgs.Show_train_info = 1000
cfgs.Smry_iter = 2000
cfgs.Total_Imgs = 133644
cfgs.ImgSize = 320
cfgs.Pkl_Path = '/data/train_record/voc_coco.pkl'
cfgs.ModelPrefix = 'refinedet320'
cfgs.Momentum = 0.9
cfgs.Weight_decay = 5e-4
cfgs.lr_steps = [80000, 100000, 120000]
cfgs.lr_gamma = 0.1
cfgs.epoch_num = 120000
#*******************************************************************************default-box
cfgs.VGG_Base = {'320': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',512, 512, 512],
            '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',512, 512, 512]}
cfgs.Extras = {'320': [256, 'S', 512],'512': [256, 'S', 512]}
# number of boxes per feature map location (0.5,1,2)
cfgs.Scale_num = {'320': [3, 3, 3, 3], '512': [3, 3, 3, 3]}  
cfgs.Tcb = {'320': [512, 512, 1024, 512],'512': [512, 512, 1024, 512]}
cfgs.PriorBox_Cfg = { '320': {'feature_maps': [40, 20, 10, 5],
                                'steps': [8, 16, 32, 64],
                                'min_sizes': [32, 64, 128, 256],
                                'max_sizes': [],
                                'aspect_ratios': [[2], [2], [2], [2]],
                                'variance': [0.1, 0.2],
                                'clip': True,
                                'name': 'RefineDet_VOC_320'},
    '512': {'feature_maps': [64, 32, 16, 8],
            'steps': [8, 16, 32, 64],
            'min_sizes': [32, 64, 128, 256],
            'max_sizes': [],
            'aspect_ratios': [[2], [2], [2], [2]],
            'variance': [0.1, 0.2],
            'clip': True,
            'name': 'RefineDet_VOC_320'}}

