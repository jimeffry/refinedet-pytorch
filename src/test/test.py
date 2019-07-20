from __future__ import print_function
#
import os
import sys
import numpy as np
import glob
import cv2
import time
import argparse
import torch
from torch.autograd import Variable
sys.path.append(os.path.join(os.path.dirname(__file__),'../utils'))
from detection_refinedet import Detect_RefineDet
sys.path.append(os.path.join(os.path.dirname(__file__),'../networks'))
#from refinedet import build_refinedet
from refinedet_train_test import build_refinedet
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfgs
#
parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--weights', default='/data/models/refinedet/RefineDet320_VOC_62000.pth',
                    type=str, help='Trained state_dict file path')
parser.add_argument('--cuda', default=False, type=bool,
                    help='Use cuda in live demo')
parser.add_argument('--img_path',type=str,default='',help='')
args = parser.parse_args()

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX


def cv2_demo(net, transform,imgpath):
    def predict(frame):
        height, width = frame.shape[:2]
        x = torch.from_numpy(transform(frame)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))
        t1=time.time()
        y = net(x)  # forward pass
        detections = y.data
        t2=time.time()
        print('consume:',t2-t1)
        # scale each detection back up to the image
        scale = torch.Tensor([width, height, width, height])
        for i in range(1,detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.1:
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                cv2.rectangle(frame,
                              (int(pt[0]), int(pt[1])),
                              (int(pt[2]), int(pt[3])),
                              COLORS[i % 3], 2)
                #labelmap[i - 1]
                score = labelmap[i - 1]+"_%.2f" %(detections[0, i, j, 0].data.numpy()) 
                cv2.putText(frame,score , (int(pt[0]), int(pt[1])),
                            FONT, 2, (255, 255, 255), 2, cv2.LINE_AA)
                j += 1
        return frame

    # start video stream thread, allow buffer to fill
    print("[INFO] starting threaded video stream...")
    # start fps timer
    # loop over frames from the video file stream
    if os.path.isdir(imgpath):
        img_paths = glob.glob(os.path.join(imgpath,'*'))
        for idx,tmp in enumerate(img_paths):
            img = cv2.imread(tmp)
            if img is None:
                print('None',tmp)
                continue
            frame = predict(img)
            cv2.imshow('result',frame)
            cv2.waitKey(100)
            savepath = os.path.join(imgpath,'test_%d.jpg' % idx)
            cv2.imwrite(savepath,frame)
    elif os.path.isfile(imgpath):
        img = cv2.imread(imgpath)
        if img is not None:
            # grab next frame
            # update FPS counter
            frame = predict(img)
            # keybindings for display
            cv2.imshow('result',frame)
            key = cv2.waitKey(0) 


if __name__ == '__main__':
    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

    from data import BaseTransform, VOC_CLASSES as labelmap
    #from ssd import build_ssd
    sys.path.append(path.join(path.dirname(__file__),'../models'))
    from refinedet import build_refinedet
    #num_classes = len(labelmap) + 1                      # +1 for background
    net = build_refinedet('test', 320, 9)            # initialize SSD
    net.load_state_dict(torch.load(args.weights,map_location='cpu'))
    net.eval()
    #net = build_ssd('test', 300, 21)    # initialize SSD
    #net.load_state_dict(torch.load(args.weights))(104/255.0, 117/255.0, 123/255.0)
    #net.size
    transform = BaseTransform(320,(104/255.0, 117/255.0, 123/255.0) )
    cv2_demo(net, transform,args.img_path)
    # stop the timer and display FPS information
    cv2.destroyAllWindows()
