import numpy as np
import os
import sys
import torch
from math import sqrt as sqrt
from itertools import product as product
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfgs


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfgs.ImgSize
        self.variance = cfg['variance'] 
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        self.version = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        '''
        every point in feature map, generate 3 default boxes: w/h = 0.5, 1, 2
        '''
        default_boxes = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]
                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k
                '''
                cx = j * self.steps[k]
                cy = i * self.steps[k]
                cx = np.minimum(np.maximum(0.0,cx),self.image_size)
                cy = np.minimum(np.maximum(0.0,cy),self.image_size)
                '''
                # aspect_ratio: 1
                # rel size: min_size
                s_k = self.min_sizes[k] /float(self.image_size)
                default_boxes += [cx, cy, s_k, s_k]
                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                if len(self.max_sizes):
                    #s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                    s_k_prime = self.max_sizes[k]/float(self.image_size)
                    default_boxes += [cx, cy, s_k_prime, s_k_prime]
                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    w = s_k*sqrt(ar)
                    #w = np.minimum(np.maximum(0,w),self.image_size)
                    h = s_k/sqrt(ar)
                    w_ = s_k_prime * sqrt(ar)
                    h_ = s_k_prime / sqrt(ar)
                    #h = np.minimum(np.maximum(0,h),self.image_size)
                    default_boxes += [cx, cy, w, h]
                    default_boxes += [cx, cy, h, w]
                    default_boxes += [cx, cy, w_,h_]
                    default_boxes += [cx,cy,h_,w_]
        # back to torch land
        #mean_p = np.array(default_boxes)
        #np.savetxt('test.txt',mean_p)
        default_output = torch.Tensor(default_boxes).view(-1, 4)
        if self.clip:
            default_output.clamp_(max=1, min=0)
        return default_output
        
if __name__=='__main__':
    voc_refinedet = {
        'num_classes': 21,
        'lr_steps': (80000, 100000, 120000),
        'max_iter': 120000,
        'feature_maps': [40, 20, 10, 5],
        'min_dim': 320,
        'steps': [8, 16, 32, 64],
        'min_sizes': [32, 64, 128, 256],
        'max_sizes': [],
        'aspect_ratios': [[2], [2], [2], [2]],
        'variance': [0.1, 0.2],
        'clip': True,
        'name': 'RefineDet_VOC_320'
    }
    priorbox = PriorBox(voc_refinedet)
    priors = priorbox.forward()
    #6375,4
    print(priors.size())
    print(priors[:10])