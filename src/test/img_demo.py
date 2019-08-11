#author: lxy
#time: 14:30 2019.7.6
#tool: python3
#version: 0.1
#project: pedestrian detection
#modify:
#####################################################
import os
import sys
import numpy as np
import glob
import cv2
import time
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
sys.path.append(os.path.join(os.path.dirname(__file__),'../utils'))
from detection_refinedet import Detect_RefineDet
sys.path.append(os.path.join(os.path.dirname(__file__),'../networks'))
from refinedet import build_refinedet
#from refinedet_resnet import build_refinedet
#from refinedet_train_test import build_refinedet
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfgs

def parms():
    parser = argparse.ArgumentParser(description='refinedet test')
    parser.add_argument('--weights', default='/data/models/refinedet/RefineDet320_VOC_115000.pth',
                        type=str, help='Trained state_dict file path')
    parser.add_argument('--cuda', default=False, type=bool,
                        help='Use cuda in live demo')
    parser.add_argument('--img_path',type=str,default='',help='')
    parser.add_argument('--load_num',type=int,default=0,help='load model num')
    parser.add_argument('--img_dir',type=str,default='',help='')
    parser.add_argument('--save_dir',type=str,default='',help='')
    return parser.parse_args()

class Refinedet_test(object):
    def __init__(self,args):
        self.img_size = cfgs.ImgSize
        self.img_dir = args.img_dir
        self.save_dir = args.save_dir
        self.build_net()
        self.load_model(args.load_num)
    
    def build_net(self):
        #self.RefineDet_model = build_refinedet('test',320,9)
        self.RefineDet_model = build_refinedet()
        self.Detector = Detect_RefineDet()

    def load_model(self,load_num):
        load_path = "%s/%s_%s.pth" %(cfgs.model_dir,cfgs.ModelPrefix,load_num)
        print('Resuming training, loading {}...'.format(load_path))
        if torch.cuda.is_available():
            self.RefineDet_model.load_weights(load_path)
            self.RefineDet_model.cuda()
            #self.Detector.cuda()
        else: 
            self.RefineDet_model.load_state_dict(torch.load(load_path,map_location='cpu'))
        self.RefineDet_model.eval()
    
    def inference(self,input_img):
        '''
        input_img: [batch,c,h,w]
        '''
        arm_loc_data, arm_conf_data, odm_loc_data, odm_conf_data, priors,odm_conf_maps = self.RefineDet_model(input_img)
        rectangles = self.Detector(arm_loc_data, arm_conf_data, odm_loc_data, odm_conf_data, priors)
        #rectangles = self.RefineDet_model(input_img)
        return rectangles.data,odm_conf_maps

    def re_scale(self,img):
        img_h, img_w = img.shape[:2]
        ratio = max(img_h, img_w) / float(self.img_size)
        new_h = int(img_h / ratio)
        new_w = int(img_w / ratio)
        ox = (self.img_size - new_w) // 2
        oy = (self.img_size - new_h) // 2
        #scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        #out = np.ones((self.img_size, self.img_size, 3), dtype=np.uint8) * 127
        #out[oy:oy + new_h, ox:ox + new_w, :] = scaled
        
        out = cv2.resize(img, (self.img_size, self.img_size))
        '''
        boxes[:,0] = boxes[:,0] * new_w + ox
        boxes[:,1] = boxes[:,1] * new_h + oy
        boxes[:,2] = boxes[:,2] * new_w + ox 
        boxes[:,3] = boxes[:,3] * new_h + oy 
        '''
        return out.astype(np.float32),[new_h,new_w,oy,ox]

    def de_scale(self,box,new_h,new_w,oy,ox,img_h,img_w):
        xmin, ymin, xmax, ymax = box[:,:,:,1],box[:,:,:,2],box[:,:,:,3],box[:,:,:,4]
        '''
        xmin = np.maximum(np.minimum(xmin*self.img_size,self.img_size),0)
        xmax = np.maximum(np.minimum(xmax*self.img_size,self.img_size),0)
        ymin = np.maximum(np.minimum(ymin*self.img_size,self.img_size),0)
        ymax = np.maximum(np.minimum(ymax*self.img_size,self.img_size),0)
        box[:,:,:,1] = (xmin - ox) / float(new_w) * img_w
        box[:,:,:,2] = (ymin - oy) / float(new_h) * img_h
        box[:,:,:,3] = (xmax - ox) / float(new_w) * img_w
        box[:,:,:,4] = (ymax - oy) / float(new_h) * img_h
        '''
        box[:,:,:,1] = np.minimum(np.maximum(xmin * img_w,0),img_w)
        box[:,:,:,2] = np.minimum(np.maximum(ymin * img_h,0),img_h)
        box[:,:,:,3] = np.minimum(np.maximum(xmax * img_w,0),img_w)
        box[:,:,:,4] = np.minimum(np.maximum(ymax * img_h,0),img_h)
        
        return box
    
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
        
        #img[:,:,0] -= cfgs.PIXEL_MEAN[0]
        #img[:,:,1] -= cfgs.PIXEL_MEAN[1]
        #img[:,:,2] -= cfgs.PIXEL_MEAN[2]
        mean = np.array(cfgs.PIXEL_MEAN,dtype=np.float32)
        img -= mean
        #img = (img-127.5)/128.0
        
        return img.astype(np.float32)

    def label_show(self,boxes,frame):
        height,width = frame.shape[:2]
        scale = np.array([width, height, width, height])
        COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        for i in range(1,boxes.shape[1]):
            j = 0
            while boxes[0, i, j, 0] >= cfgs.odm_threshold:
                pt = boxes[0, i, j, 1:] #* scale
                #print(pt)
                cv2.rectangle(frame,
                              (int(pt[0]), int(pt[1])),
                              (int(pt[2]), int(pt[3])),
                              (0,255,0), 2)
                cv2.putText(frame, cfgs.shownames[i], (int(pt[0]), int(pt[1])),
                            FONT, 1, (255, 255, 255), 1, 4)#cv2.LINE_AA)
                j += 1

    def test_img(self,frame):
        height, width = frame.shape[:2]
        img_scale, window = self.re_scale(frame.copy())
        #img_scale = self.normalize(img_scale)
        img_input = torch.from_numpy(img_scale).permute(2, 0, 1)
        img_input = Variable(img_input.unsqueeze(0))
        if torch.cuda.is_available():
            img_input = img_input.cuda()
        t1=time.time()
        rectangles,conf_maps = self.inference(img_input)  # forward pass
        detections = rectangles.cpu().numpy()
        t2=time.time()
        print('consume:',t2-t1)
        # scale each detection back up to the image
        detections = self.de_scale(detections,window[0],window[1],window[2],window[3],height,width)
        self.label_show(detections,frame)
        return frame,conf_maps

    def get_hotmaps(self,conf_maps):
        '''
        conf_maps: feature_pyramid maps for classification
        '''
        hotmaps = []
        for tmp_map in conf_maps:
            batch,h,w,c = tmp_map.size()
            tmp_map = tmp_map.view(batch,h,w,-1,cfgs.ClsNum)
            tmp_map = tmp_map[0,:,:,:,1:]
            tmp_map_soft = torch.nn.functional.softmax(tmp_map,dim=3)
            cls_mask = torch.argmax(tmp_map_soft,dim=3,keepdim=True)
            #score,cls_mask = torch.max(tmp_map_soft,dim=4,keepdim=True)
            #cls_mask = cls_mask.unsqueeze(4).expand_as(tmp_map_soft)
            #print(cls_mask.data.size(),tmp_map_soft.data.size())
            tmp_hotmap = tmp_map_soft.gather(3,cls_mask)
            map_mask = torch.argmax(tmp_hotmap,dim=2,keepdim=True)
            tmp_hotmap = tmp_hotmap.gather(2,map_mask)
            tmp_hotmap.squeeze_(3)
            tmp_hotmap.squeeze_(2)
            print('map max:',tmp_hotmap.data.max())
            hotmaps.append(tmp_hotmap.data.numpy())
        return hotmaps

    def display_hotmap(self,hotmaps):
        '''
        hotmaps: a list of hot map ,every shape is [1,h,w]
        '''       
        fig, axes = plt.subplots(nrows=2, ncols=2, constrained_layout=True)
        ax1 = axes[0,0]
        im1 = ax1.imshow(hotmaps[0])
        # We want to show all ticks...
        #ax.set_xticks(np.arange(len(farmers)))
        #ax.set_yticks(np.arange(len(vegetables)))
        # ... and label them with the respective list entries
        #ax.set_xticklabels(farmers)
        #ax.set_yticklabels(vegetables)
        # Rotate the tick labels and set their alignment.
        #plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         #       rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        #for i in range(len(vegetables)):
         #   for j in range(len(farmers)):
          #      text = ax.text(j, i, harvest[i, j],
           #                 ha="center", va="center", color="w")
        #cb1 = fig.colorbar(im1)
        ax1.set_title("feature_3")
        #**************************************************************
        ax2 = axes[0,1]
        im2 = ax2.imshow(hotmaps[1])
        #cb2 = fig.colorbar(im2)
        ax2.set_title('feature_4')
        #************************************************
        ax3 = axes[1,0]
        im3 = ax3.imshow(hotmaps[2])
        #cb3 = fig.colorbar(im3)
        ax3.set_title('feature_5')
        #**********************************************
        img = hotmaps[3]
        min_d = np.min(img)
        max_d = np.max(img)
        tick_d = []
        while min_d < max_d:
            tick_d.append(min_d)
            min_d+=0.01
        ax4 = axes[1,1]
        im4 = ax4.imshow(hotmaps[3])
        cb4 = fig.colorbar(im4) #ticks=tick_d)
        ax4.set_title('feature_6')
        #fig.tight_layout()
        plt.savefig('hotmap.png')
        plt.show()

    def test_dir(self,imgpath):
        print(imgpath)
        if os.path.isdir(imgpath):
            img_paths = glob.glob(os.path.join(imgpath,'*'))
            save_dir = os.path.join(imgpath,'test')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            for idx,tmp in enumerate(img_paths):
                if not os.path.isfile(tmp):
                    continue
                img = cv2.imread(tmp)
                if img is None:
                    print('None',tmp)
                    continue
                frame,_ = self.test_img(img)
                cv2.imshow('result',frame)
                cv2.waitKey(100)
                savepath = os.path.join(save_dir,'test_%d.jpg' % idx)
                cv2.imwrite(savepath,frame)
        elif os.path.isfile(imgpath) and imgpath.endswith('txt'):
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            f_r = open(imgpath,'r')
            file_cnts = f_r.readlines()
            for j in tqdm(range(len(file_cnts))):
                tmp_file = file_cnts[j].strip()
                #tmp_file = file_cnts[j].strip()+'.jpg'
                tmp_splits = tmp_file.split(',')
                tmp_file = tmp_splits[0] #.split('/')[-1]
                gt_box = map(float,tmp_splits[1:])
                gt_box = np.array(list(gt_box))
                gt_box = gt_box.reshape([-1,5])
                save_name = tmp_file
                tmp_path = os.path.join(self.img_dir,tmp_file) 
                if not os.path.exists(tmp_path):
                    continue
                img = cv2.imread(tmp_path) 
                if img is None:
                    print('None',tmp)
                    continue
                #frame,_ = self.test_img(img)                
                for idx in range(gt_box.shape[0]):
                    pt = gt_box[idx,:4]
                    i = int(gt_box[idx,4])
                    cv2.rectangle(img,
                                (int(pt[0]), int(pt[1])),
                                (int(pt[2]), int(pt[3])),
                                (0,0,255), 2) 
                    cv2.putText(img, cfgs.VOCDataNames[i], (int(pt[0]), int(pt[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, 4)#cv2.LINE_AA)
                cv2.imshow('result',img)
                cv2.waitKey(0)               
                #savepath = os.path.join(self.save_dir,save_name)
                #cv2.imwrite(savepath,frame)
        elif os.path.isfile(imgpath) and imgpath.endswith(('.mp4','.avi')) :
            #url = "rtsp://admin:dh123456@192.168.2." + ip + "/cam/realmonitor?channel=1&subtype=0"
            url = "rtsp://admin:hk123456@192.168.1.64/h264/1/main/av_stream"
            cap = cv2.VideoCapture(imgpath)
            if not cap.isOpened():
                print("failed open camera")
                return 0
            else: 
                while cap.isOpened():
                    _,img = cap.read()
                    frame,_ = self.test_img(img)
                    cv2.imshow('result',frame)
                    q=cv2.waitKey(10) & 0xFF
                    if q == 27 or q ==ord('q'):
                        break
            cap.release()
            cv2.destroyAllWindows()
        elif os.path.isfile(imgpath):
            img = cv2.imread(imgpath)
            if img is not None:
                # grab next frame
                # update FPS counter
                frame,odm_maps = self.test_img(img)
                hotmaps = self.get_hotmaps(odm_maps)
                self.display_hotmap(hotmaps)
                # keybindings for display
                cv2.imshow('result',frame)
                cv2.imwrite('test1.jpg',frame)
                key = cv2.waitKey(0) 
        else:
            print('please input the right img-path')

if __name__ == '__main__':
    args = parms()
    model_net = Refinedet_test(args)
    img_path = args.img_path
    model_net.test_dir(img_path)
