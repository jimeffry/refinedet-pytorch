#author: lxy
#time: 14:30 2019.6.27
#tool: python3
#version: 0.1
#project: pedestrian detection
#modify:
#####################################################
import os 
import sys
if sys.version_info[0] == 2:
    import cPickle as Pickle
    #sys.path.insert(0,'/home/lxy/anaconda3/envs/py2/lib/python2.7/site-packages')
else:
    #import _pickle as Pickle
    import pickle as Pickle
import numpy as np 
import argparse
import random
from collections import defaultdict
import cv2
from  tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__),'../utils'))
from transform import Transform
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfgs

def parms():
    parser = argparse.ArgumentParser(description='dataset convert')
    parser.add_argument('--voc-dir',dest='voc_dir',type=str,default='../../data/',\
                        help='dataset root')
    parser.add_argument('--coco-dir',dest='coco_dir',type=str,default=None,\
                        help='coco files dir')
    parser.add_argument('--image-dir',dest='image_dir',type=str,default='VOC_JPG',\
                        help='images saved dir')
    parser.add_argument('--save-dir',dest='save_dir',type=str,default='../../data/',\
                        help='tfrecord save dir')
    parser.add_argument('--save-name',dest='save_name',type=str,\
                        default='train',help='image for train or test')
    parser.add_argument('--voc-anno',dest='voc_anno',type=str,\
                        default=None,help='file2')
    parser.add_argument('--dataset-name',dest='dataset_name',type=str,default='VOC',\
                        help='datasetname')
    #for widerface
    parser.add_argument('--coco-anno',dest='coco_anno',type=str,\
                        default='../../data/wider_gt.txt',help='annotation files')
    parser.add_argument('--out-file',dest='out_file',type=str,\
                        default=None,help='datasetname')
    parser.add_argument('--record-file',dest='record_file',type=str,\
                        default='record_train.txt',help='record file')                    
    return parser.parse_args()

class convert_to_pkl(object):
    def __init__(self,args):
        self.save_dir = args.save_dir
        self.voc_dir = args.voc_dir
        self.coco_dir = args.coco_dir
        self.file_out = args.out_file
        self.voc_anno_file = args.voc_anno
        self.coco_anno_file = args.coco_anno
        self.record_file = args.record_file
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.load_files()

    def load_files(self):
        cache_file = os.path.join(self.save_dir,self.file_out)
        self.file_w = open(cache_file, 'wb') 
        self.voc_anno_r = open(self.voc_anno_file,'r')
        self.coco_anno_r = open(self.coco_anno_file,'r')
        self.record_w = open(self.record_file,'w')
        auger_list=["Sequential", "Fliplr","Dropout", \
                    "AdditiveGaussianNoise","SigmoidContrast","Multiply"]
        self.transfrom_imgs = Transform(img_auger_list=auger_list)

    def rd_anotation(self,annotation,image_dir,cnt_dict):
        '''
        annotation: 1/img_01 x1 y1 x2 y2 x1 y1 x2 y2 ...
        '''
        img_dict = dict()
        annotation = annotation.strip().split(',')
        img_prefix = annotation[0]
        #boxed change to float type
        bbox = map(float, annotation[1:])
        if not isinstance(bbox,list):
            bbox = list(bbox)
        gt_box_labels = np.asarray(bbox,dtype=np.int32).reshape(-1,5)
        #load image
        img_path = os.path.join(image_dir,img_prefix)
        if not os.path.exists(img_path):
            print('not exist:',img_path)
            return None
        img_org = cv2.imread(img_path)
        if img_org is None:
            return None
        img_org = np.array(img_org,dtype=np.uint8)
        img_dict['img_data'] = img_org
        img_dict['gt'] = gt_box_labels #gt_list
        for i in range(gt_box_labels.shape[0]):
            tmp_key = cfgs.VOCDataNames[int(gt_box_labels[i,4])]
            cnt_dict[tmp_key]+=1
        return img_dict

    def transform_imgbox(self,img_dict,cnt_dict):
        '''
        annotation: 1/img_01 x1 y1 x2 y2 x1 y1 x2 y2 ...
        '''
        #img_dict = dict()
        if img_dict is None:
            return None
        img_org = img_dict['img_data']
        gt_box_labels = img_dict['gt']
        boxes = gt_box_labels[:,:4]
        labels = gt_box_labels[:,4]
        if img_org is None:
            print("aug img is None")
            return None
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

    def write_pkl(self): 
        voc_cnts = self.voc_anno_r.readlines()
        coco_cnts = self.coco_anno_r.readlines()
        total_coco = len(coco_cnts)
        total_voc = len(voc_cnts)
        cnt_w = max(total_coco,total_voc)
        instance_cnt_dic = defaultdict(lambda:0)
        total_img_cnt = 0
        cnt_failed = 0
        for idx in tqdm(range(cnt_w)):
            if idx < total_voc:
                tmp_voc = voc_cnts[idx]
                tmp_dict = self.rd_anotation(tmp_voc,self.voc_dir,instance_cnt_dic) 
                if tmp_dict is not None:
                    Pickle.dump(tmp_dict,self.file_w,Pickle.HIGHEST_PROTOCOL)
                    total_img_cnt +=1
                    #label_show(tmp_dict)
                if random.randint(0, 1) :
                    img_dict = self.transform_imgbox(tmp_dict,instance_cnt_dic)
                    if img_dict is None:
                        cnt_failed+=1
                    else:
                        Pickle.dump(img_dict,self.file_w,Pickle.HIGHEST_PROTOCOL)
                        #label_show(img_dict)
                        total_img_cnt+=1
            tmp_coco = coco_cnts[idx]
            tmp_dict = self.rd_anotation(tmp_coco,self.coco_dir,instance_cnt_dic)
            if tmp_dict is not None:
                Pickle.dump(tmp_dict,self.file_w,Pickle.HIGHEST_PROTOCOL)
                total_img_cnt +=1
                #label_show(tmp_dict)
            if random.randint(0, 1) :
                img_dict = self.transform_imgbox(tmp_dict,instance_cnt_dic)
                if img_dict is None:
                    cnt_failed+=1
                else:
                    Pickle.dump(img_dict,self.file_w,Pickle.HIGHEST_PROTOCOL)
                    #label_show(img_dict)
                    total_img_cnt+=1
        for tmp_key in sorted(instance_cnt_dic.keys()):
            self.record_w.write("{}:{}\n".format(tmp_key,instance_cnt_dic[tmp_key]))
        self.voc_anno_r.close()
        self.coco_anno_r.close()
        self.file_w.close()
        self.record_w.close()
        print("total img:",total_img_cnt)
        print('failed aug img:',cnt_failed)

def label_show(img_dict,mode='rgb'):
    img = img_dict['img_data']
    if mode == 'rgb':
        img = img[:,:,::-1]
    img = np.array(img,dtype=np.uint8)
    gt = img_dict['gt']
    num_obj = gt.shape[0]
    #print("img",img.shape)
    #print("box",gt.shape)
    for i in range(num_obj):
        #for rectangle in gt:
        rectangle = gt[i]
        #print('show bbl',rectangle)
        #print(map(int,rectangle[5:]))
        score_label = str("{}".format(rectangle[4]))
        #score_label = str(1.0)
        cv2.putText(img,score_label,(int(rectangle[0]),int(rectangle[1])),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0))
        cv2.rectangle(img,(int(rectangle[0]),int(rectangle[1])),(int(rectangle[2]),int(rectangle[3])),(255,0,0),1)
        if len(rectangle) > 5:
            for i in range(5,15,2):
                cv2.circle(img,(int(rectangle[i+0]),int(rectangle[i+1])),2,(0,255,0))
    cv2.imshow("img",img)
    cv2.waitKey(0)
                
if __name__=='__main__':
    args = parms()
    pkl_w = convert_to_pkl(args)
    pkl_w.write_pkl()               