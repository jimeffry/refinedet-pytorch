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
#cfgs.VOCDataNames = ['aeroplane', 'bicycle', 'bird', 'boat','bottle', 'bus', 'car', 'cat', 'chair',
 #   'cow', 'diningtable', 'dog', 'horse','motorbike', 'person', 'pottedplant','sheep', 'sofa', 'train', 'tvmonitor']
#cfgs.PIXEL_MEAN = [0.485,0.456,0.406] # R, G, B
cfgs.PIXEL_MEAN = [104,117,123]
cfgs.PIXEL_NORM = [0.229,0.224,0.225] #rgb
cfgs.variance = [0.1, 0.2]
cfgs.voc_file = '/home/lxy/Develop/git_prj/refinedet-pytorch/datas/VOC/test_voc07.txt' #'/home/lixiaoyu/Develop/refinedet-pytorch/datas/train_voc.txt'  
cfgs.voc_dir = '/data/VOC/VOCdevkit' #'/wdc/LXY.data/VOC/VOCdevkit' #
cfgs.coco_file = '/home/lixiaoyu/Develop/refinedet-pytorch/datas/coco2017.txt'
cfgs.coco_dir = '/wdc/LXY.data/CoCo2017'
#**********************************************************************train
cfgs.Show_train_info = 100
cfgs.Smry_iter = 2000
cfgs.Total_Imgs = 133459#133644
cfgs.ImgSize = 320
cfgs.Pkl_Path = '/data/train_record/voc_coco.pkl'
cfgs.ModelPrefix = 'resnet320' #'refinedet320' #'RefineDet320_VOC'
cfgs.Momentum = 0.9
cfgs.Weight_decay = 5e-4
cfgs.lr_steps = [20000, 40000, 60000]
cfgs.lr_gamma = 0.1
cfgs.epoch_num = 120000
#*******************************************************************************default-box
cfgs.VGG_Base = {'320': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',512, 512, 512],
            '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',512, 512, 512]}
cfgs.Extras = {'320': [256, 'S', 512],'512': [256, 'S', 512]}
cfgs.feature_map_filters = [512,1024,2048,2048]
# number of boxes per feature map location (0.5,1,2)
cfgs.Scale_num = {'320': [6, 6, 6, 6], '512': [6, 6, 6, 6]}  #6, 6, 6, 6
cfgs.Tcb = {'320': [512, 512, 1024, 512],'512': [512, 512, 1024, 512]}
cfgs.PriorBox_Cfg = { '320': {'feature_maps': [40, 20, 10, 5],
                                'steps': [8, 16, 32, 64],
                                'min_sizes': [25,48, 105, 256], #[25,48, 105, 256],
                                'max_sizes': [48,105,163,278], #[32,64,128,256], #[48,105,163,278],
                                'aspect_ratios': [[2,3], [2,3], [2,3], [2,3]],
                                'variance': [0.1, 0.2],
                                'clip': True,
                                'name': 'RefineDet_VOC_320'},
    '512': {'feature_maps': [64, 32, 16, 8],
            'steps': [8, 16, 32, 64],
            'min_sizes': [16, 64, 128, 256], #[25,48, 105, 256],
            'max_sizes': [64,128,256,500] ,#[48,105,163,500],
            'aspect_ratios': [[2], [2], [2], [2]],
            'variance': [0.1, 0.2],
            'clip': True,
            'name': 'RefineDet_VOC_512'}}
cfgs.PriorBox_Cfg_resnet = {'320': {'feature_maps': [40,20,10,5],
                                'steps': [8, 16, 32, 64],
                                'min_sizes': [16, 32, 64, 128],
                                'max_sizes': [32,64,128,256],
                                'aspect_ratios': [[2], [2], [2], [2]],
                                'variance': [0.1, 0.2],
                                'clip': True,
                                'name': 'RefineDet_VOC_320'},
                            '512': {'feature_maps': [32, 16, 8, 4],
                                'steps': [8, 16, 32, 64],
                                'min_sizes': [32, 64, 128, 256],
                                'max_sizes': [],
                                'aspect_ratios': [[2], [2], [2], [2]],
                                'variance': [0.1, 0.2],
                                'clip': True,
                                'name': 'RefineDet_VOC_320'}}

#[[2,3], [2,3], [2,3], [2,3]],[25,48, 105, 256],[48,105,163,278],[[2], [2], [2], [2]],[32, 64, 128, 256],
#************************************************************************************test
cfgs.top_k = 1000
cfgs.odm_threshold = 0.95
cfgs.nms_threshold = 0.5 #0.45
cfgs.arm_threshold = 0.1
cfgs.model_dir = '/data/models/refinedet'
cfgs.shownames = ['bg','person','bicycle','motorbike','car','bus','aeroplane','train','boat']
#cfgs.shownames = ['bg','aeroplane', 'bicycle', 'bird', 'boat','bottle', 'bus', 'car', 'cat', 'chair',
#    'cow', 'diningtable', 'dog', 'horse','motorbike', 'person', 'pottedplant','sheep', 'sofa', 'train', 'tvmonitor']
