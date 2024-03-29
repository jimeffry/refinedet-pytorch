#author: lxy
#time: 14:30 2019.7.1
#tool: python3
#version: 0.1
#project: pedestrian detection
#modify:
#####################################################
import os
import sys
if sys.version_info[0] == 2:
    import cPickle as Pickle
    sys.path.insert(0,'/home/lxy/anaconda3/envs/py2/lib/python2.7/site-packages')
else:
    #import _pickle as Pickle
    import pickle as Pickle
import torch
import cv2
import numpy as np
import torch.utils.data as u_data
#from convert_to_pickle import label_show
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfgs

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations
    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets


class ReadPkDataset(u_data.Dataset): #data.Dataset
    """
    VOC Detection Dataset Object
    """
    def __init__(self):
        self.total_imgs = cfgs.Total_Imgs
        self.Pickle_path = cfgs.Pkl_Path
        self.img_size = cfgs.ImgSize
        self.load_pkl()
        self.idx = 0

    def __getitem__(self, index):
        im, gt = self.pull_item(index)
        return im, gt

    def __len__(self):
        return self.total_imgs

    def load_pkl(self):
        self.pkl_load = open(self.Pickle_path,'rb')
    def close_pkl(self):
        self.pkl_load.close()

    def get_batch(self,batch_size):
        batch_data = torch.zeros([batch_size,3,self.img_size,self.img_size],dtype=torch.float32)
        targets = []
        for tmp_idx in range(batch_size):
            img,gt = self.pull_item(self.idx)
            self.idx +=1
            batch_data[tmp_idx,:,:,:] = img
            targets.append(torch.FloatTensor(gt))
        return batch_data,targets

    def pull_item(self, index):
        '''
        output: img - shape(c,h,w)
                gt_boxes+label: box-(x1,y1,x2,y2)
                label: dataset_class_num 
        '''
        tmp_dict = Pickle.load(self.pkl_load)
        img_data = tmp_dict['img_data']
        img_data = img_data[:,:,::-1]
        gt_box_label = tmp_dict['gt']
        #print(gt_box_label)
        #gt_box_label[:,-1] = gt_box_label[:,-1] +1
        img_data, gt_box_label = self.re_scale(img_data,gt_box_label)
        img_data = self.normalize(img_data)
        if index == self.total_imgs-1:
            self.close_pkl()
            self.load_pkl()
        return torch.from_numpy(img_data).permute(2, 0, 1),gt_box_label
        #return img_data,gt_box_label
    
    def re_scale(self,img, boxes):
        img_h, img_w = img.shape[:2]
        boxes = np.array(boxes,dtype=np.float32)
        '''
        ratio = max(img_h, img_w) / float(self.img_size)
        new_h = int(img_h / ratio)
        new_w = int(img_w / ratio)
        ox = (self.img_size - new_w) // 2
        oy = (self.img_size - new_h) // 2
        scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        out = np.ones((self.img_size, self.img_size, 3), dtype=np.uint8) * 127
        out[oy:oy + new_h, ox:ox + new_w, :] = scaled
        '''
        out = cv2.resize(img, (self.img_size, self.img_size))
        boxes[:,0] = boxes[:,0] / float(img_w)
        boxes[:,1] = boxes[:,1] / float(img_h)
        boxes[:,2] = boxes[:,2] / float(img_w)
        boxes[:,3] = boxes[:,3] / float(img_h)
        '''
        boxes[:,0] = boxes[:,0] * new_w + ox
        boxes[:,1] = boxes[:,1] * new_h + oy
        boxes[:,2] = boxes[:,2] * new_w + ox 
        boxes[:,3] = boxes[:,3] * new_h + oy 
        '''
        return out, boxes
    
    def normalize(self,img):
        '''
        img = img / 255.0
        img[:,:,0] -= cfgs.PIXEL_MEAN[0]
        img[:,:,0] = img[:,:,0] / cfgs.PIXEL_NORM[0] 
        img[:,:,1] -= cfgs.PIXEL_MEAN[1]
        img[:,:,1] = img[:,:,1] / cfgs.PIXEL_NORM[1]
        img[:,:,2] -= cfgs.PIXEL_MEAN[2]
        img[:,:,2] = img[:,:,2] / cfgs.PIXEL_NORM[2]
        '''
        img[:,:,0] -= cfgs.PIXEL_MEAN[0]
        img[:,:,1] -= cfgs.PIXEL_MEAN[1]
        img[:,:,2] -= cfgs.PIXEL_MEAN[2]
        return img.astype(np.float32)

if __name__=='__main__':
    test_d = ReadPkDataset()
    img_dict = dict()
    i=0
    total = 133644
    while 3-i:
        img, gt = test_d.get_batch(2)
        img_dict['img_data'] = img[0].numpy()
        img_dict['gt'] = gt[0]
        #label_show(img_dict)
        print(gt[0][:,-1])
        #sys.stdout.write('\r>> %d /%d' %(i,total))
        #sys.stdout.flush()
        i+=1
    print(i)