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
import random
import torch.utils.data as u_data
#from convert_to_pickle import label_show
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfgs
sys.path.append(os.path.join(os.path.dirname(__file__),'../utils'))
from transform import Transform

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


class ReadDataset(u_data.Dataset): #data.Dataset
    """
    VOC Detection Dataset Object
    """
    def __init__(self):
        self.voc_file = cfgs.voc_file
        self.coco_file = cfgs.coco_file
        self.img_size = cfgs.ImgSize
        self.voc_dir = cfgs.voc_dir
        self.coco_dir = cfgs.coco_dir
        self.ids = []
        self.annotations = []
        self.load_txt()
        self.idx = 0
        self.total_num = self.__len__()
        self.shulf_num = list(range(self.total_num))
        random.shuffle(self.shulf_num)
        auger_list=["Sequential", "Fliplr","Dropout", \
                    "AdditiveGaussianNoise","SigmoidContrast","Multiply"]
        self.transfrom_imgs = Transform(img_auger_list=auger_list)

    def __getitem__(self, index):
        im, gt,_,_ = self.pull_item(index)
        return im, gt

    def __len__(self):
        return len(self.annotations)

    def load_txt(self):
        self.voc_r = open(self.voc_file,'r')
        #self.coco_r = open(self.coco_file,'r')
        voc_annotations = self.voc_r.readlines()
        #coco_annotations = self.coco_r.readlines()
        for tmp in voc_annotations:
            tmp_splits = tmp.strip().split(',')
            img_path = os.path.join(self.voc_dir,tmp_splits[0])
            self.ids.append((self.voc_dir,tmp_splits[0].split('/')[-1][:-4]))
            bbox = map(float, tmp_splits[1:])
            if not isinstance(bbox,list):
                bbox = list(bbox)
            bbox.insert(0,img_path)
            self.annotations.append(bbox)
        '''
        for tmp in coco_annotations:
            tmp_splits = tmp.strip().split(',')
            img_path = os.path.join(self.coco_dir,tmp_splits[0])
            bbox = map(float, tmp_splits[1:])
            if not isinstance(bbox,list):
                bbox = list(bbox)
            bbox.insert(0,img_path)
            self.annotations.append(bbox)
        '''
    def close_txt(self):
        self.voc_r.close()
        self.coco_r.close()

    def get_batch(self,batch_size):
        batch_data = torch.zeros([batch_size,3,self.img_size,self.img_size],dtype=torch.float32)
        targets = []
        if self.idx >= self.total_num -1:
            random.shuffle(self.shulf_num)
            self.idx = 0
        for tmp_idx in range(batch_size):
            if self.idx >= self.total_num:
                rd_idx = 0
            else:
                rd_idx = self.shulf_num[self.idx]
            img,gt,_,_ = self.pull_item(rd_idx)
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
        tmp_annotation = self.annotations[index]
        tmp_path = tmp_annotation[0]
        img_data = cv2.imread(tmp_path)
        h,w = img_data.shape[:2]
        img_data = img_data[:,:,::-1]
        gt_box_label = np.array(tmp_annotation[1:],dtype=np.float32).reshape(-1,5)
        #print(gt_box_label) 
        img_data, gt_box_label = self.re_scale(img_data,gt_box_label)
        img_data = self.normalize(img_data)
        return torch.from_numpy(img_data).permute(2, 0, 1),gt_box_label,h,w
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
        boxes[:,0] = boxes[:,0] / float(img_w)
        boxes[:,1] = boxes[:,1] / float(img_h)
        boxes[:,2] = boxes[:,2] / float(img_w)
        boxes[:,3] = boxes[:,3] / float(img_h)
        out = cv2.resize(img, (self.img_size, self.img_size))
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

    def transform(self,img,gt_box_labels):
        '''
        annotation: 1/img_01 x1 y1 x2 y2 x1 y1 x2 y2 ...
        '''
        #img_dict = dict()
        if img is None:
            return None
        boxes = gt_box_labels[:,:4]
        labels = gt_box_labels[:,4]
        img_aug,boxes_aug,keep_idx = self.transfrom_imgs.aug_img_boxes(img_org,[boxes.tolist()])
        if not len(boxes_aug) >0:
            #print("aug box is None")
            return None
        img_data = np.array(img_aug[0],np.uint8)
        boxes_trans = np.array(boxes_aug[0], dtype=np.int32).reshape(-1, 4)
        label = np.array(labels[keep_idx[1][0]],dtype=np.int32).reshape(-1,1)
        gt_box_labels = np.concatenate((boxes_trans,label),axis=1)
        img_dict['img_data'] = img_data
        img_dict['gt'] = gt_box_labels #gt_list
        for i in range(gt_box_labels.shape[0]):
            tmp_key = cfgs.VOCDataNames[int(gt_box_labels[i,4])]
            cnt_dict[tmp_key]+=1
        return img_dict

if __name__=='__main__':
    test_d = ReadDataset()
    img_dict = dict()
    i=0
    total = 133644
    while 3-i:
        img, gt = test_d.get_batch(2)
        img_dict['img_data'] = img[0].numpy()
        img_dict['gt'] = gt[0]
        label_show(img_dict)
        #print(gt[0][:,-1])
        #sys.stdout.write('\r>> %d /%d' %(i,total))
        #sys.stdout.flush()
        i+=1
    print(i)
