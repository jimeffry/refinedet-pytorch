#author: lxy
#time: 14:30 2019.6.27
#tool: python3
#version: 0.1
#project: pedestrian detection
#modify:
#####################################################
import numpy as np 
import os
import sys 
from collections import defaultdict
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
import string 
#import cv2
from  tqdm import tqdm
import glob
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfgs

def parms():
    parser = argparse.ArgumentParser(description='dataset convert')
    parser.add_argument('--VOC-dir',dest='VOC_dir',type=str,default='../../data/',\
                        help='dataset root')
    parser.add_argument('--xml-dir',dest='xml_dir',type=str,default='VOC_XML',\
                        help='xml files dir')
    parser.add_argument('--image-dir',dest='image_dir',type=str,default='VOC_JPG',\
                        help='images saved dir')
    parser.add_argument('--save-dir',dest='save_dir',type=str,default='../../data/',\
                        help='tfrecord save dir')
    parser.add_argument('--save-name',dest='save_name',type=str,\
                        default='train_record',help='image for train or test')
    parser.add_argument('--file2-in',dest='file2_in',type=str,\
                        default=None,help='file2')
    parser.add_argument('--dataset-name',dest='dataset_name',type=str,default='VOC',\
                        help='datasetname')
    #for widerface
    parser.add_argument('--anno-file',dest='anno_file',type=str,\
                        default='../../data/wider_gt.txt',help='annotation files')
    parser.add_argument('--out-file',dest='out_file',type=str,\
                        default=None,help='datasetname')
    parser.add_argument('--cmd-type',dest='cmd_type',type=str,\
                        default=None,help='command type')                    
    return parser.parse_args()

def read_xml_gtbox_and_label(xml_path,img_name,cnt_dict,keep_difficult=True):
    """
    :param xml_path: the path of voc xml
    :return: a list contains gtboxes and labels, shape is [num_of_gtboxes, 5],
           and has [xmin, ymin, xmax, ymax, label] in a per row
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    img_width = None
    img_height = None
    box_list = []
    cnt_pass = 0
    #print("process image ",img_name)
    for child_of_root in root:
        if child_of_root.tag == 'filename':
            assert child_of_root.text == img_name, 'image name and xml is not the same'
        
        if child_of_root.tag == 'size':
            for child_item in child_of_root:
                if child_item.tag == 'width':
                    img_width = int(child_item.text)
                if child_item.tag == 'height':
                    img_height = int(child_item.text)
        
        if child_of_root.tag == 'object':
            label = None
            for idx,child_item in enumerate(child_of_root):
                tmp_tag = child_item.tag
                tmp_text = child_item.text 
                #print(idx,tmp_tag)
                
                if tmp_tag == 'name' and tmp_text in cfgs.VOCDataNames:
                    label = cfgs.VOCDataNames.index(tmp_text)
                    #print("label:",label,child_item.text)
                    cnt_dict[tmp_text]+=1 
                
                    #print("is not in train names: ",xml_path)
                #print(child_item.tag)
                if tmp_tag == 'difficult' and int(tmp_text)==1 and not keep_difficult:
                    cnt_pass +=1
                    print('pass difficult')
                    return img_height,img_width,box_list,cnt_pass
                #print('tag:',child_item.tag)
                if tmp_tag == 'bndbox':
                    for node in child_item:
                        if node.tag == 'xmin':
                            x1 = np.int32(float(node.text))
                        if node.tag == 'ymin':
                            y1 = np.int32(float(node.text))
                        if node.tag == 'xmax':
                            x2= np.int32(float(node.text))
                        if node.tag == 'ymax':
                            y2 = np.int32(float(node.text))
                    #assert label is not None, 'label is none, error'
                    #tmp_box.append(label)  # [x1, y1. x2, y2, label]
                    #print('label:',label)
                    if label is not None:
                        box_list.extend([x1,y1,x2,y2,label])
    return img_height, img_width, box_list,cnt_pass

def convert_voc(base_dir,file_in,file_out,name):
    '''
    base_dir: VOC root dir
    file_in: annotation file- img_name
    file_out: out annotaion- img_name, bounding-boxes-list
    '''
    annotation_f = open(file_in,'r')
    f_out = open(file_out,'w')
    record_w = open('%s.txt' % name,'w')
    tmp_f = open('voc12trainval.txt','w')
    p_cnt = 0
    cnt_none = 0
    cnt_img =0
    instance_cnt_dic = defaultdict(lambda:0)
    tmp_dir = base_dir.split('/')[-1]
    file_cnts = annotation_f.readlines()
    for i in tqdm(range(len(file_cnts))):
        #annot_splits = one_line.strip().split()
        #img_name = annot_splits[0].split('/')[-1]
        #xml_path = annot_splits[1]
        one_line = file_cnts[i]
        img_name = one_line.strip()+'.jpg'
        xml_path = one_line.strip()+'.xml'
        xml_path = os.path.join(base_dir,'Annotations',xml_path)
        h,w, gtboxes,cnt_pass = read_xml_gtbox_and_label(xml_path,img_name,instance_cnt_dic)
        p_cnt+=cnt_pass
        if len(gtboxes)<1 :
            #print("gbox is none: ",annot_splits[0])
            cnt_none+=1
            continue
        else:
            cnt_img+=1
            #img = cv2.imread(os.path.join(base_dir,'JPEGImages',img_name))
            #for box in gtboxes:
             #   cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),(0,255,0))
            #cv2.imshow("imgshow",img)
            #cv2.waitKey(1000)
            img_path = os.path.join(tmp_dir,'JPEGImages',img_name)
            gt_str = map(str,gtboxes)
            gt_str = ','.join(gt_str)
            f_out.write("{},{}\n".format(img_path,gt_str))
            tmp_f.write("{}\n".format(one_line.strip()))
    for tmp_key in sorted(instance_cnt_dic.keys()):
        record_w.write("{}:{}\n".format(tmp_key,instance_cnt_dic[tmp_key]))
    annotation_f.close()
    f_out.close()
    record_w.close()
    print("pass: %s, none: %s, img_num: %s" %(p_cnt,cnt_none,cnt_img))

def get_dir_cnt(dirpath,outfile):
    f_w = open(outfile,'w')
    paths = os.listdir(dirpath)
    #paths = glob.glob(os.path.join(dirpath, '*.xml'))
    for tmp in paths:
        if tmp.endswith("xml"):
            f_w.write("{}\n".format(tmp[:-4]))
    f_w.close

def merge_2file(file1,file2,file_out):
    '''
    file1: img_name, x1,y1,x2,y2,label,x11,y11,x22,y22,label2,...
    file2: img_name, x1,y1,x2,y2,label,x11,y11,x22,y22,label2,...
    '''
    f1_r = open(file1,'r')
    f2_r = open(file2,'r')
    f_w = open(file_out,'w')
    f1_cnts = f1_r.readlines()
    f2_cnts = f2_r.readlines()
    for tmp in f1_cnts:
        tmp_s = tmp.strip().split(',')
        tmp_s[0] = tmp_s[0] +'.jpg'
        tmp_w = ','.join(tmp_s)
        f_w.write("{}\n".format('train2017/'+tmp_w))
    for tmp in f2_cnts:
        tmp_s = tmp.strip().split(',')
        tmp_s[0] = tmp_s[0] +'.jpg'
        tmp_w = ','.join(tmp_s)
        f_w.write("{}\n".format('val2017/'+tmp_w))
    f1_r.close()
    f2_r.close()
    f_w.close()

def filter_area(file_in,file_out):
    '''
    '''
    f_r = open(file_in,'r')
    f_w = open(file_out,'w')
    f_cnts = f_r.readlines()
    for tmp_f in f_cnts:
        tmp = tmp_f.strip().split(',')
        imgpath = tmp[0]
        bbox = map(float,tmp[1:])
        bbox = np.array(list(bbox))
        bbox = bbox.reshape([-1,5])
        tmp_bb =[]
        for idx in range(bbox.shape[0]):
            pt = bbox[idx]
            w = pt[2] - pt[0]
            h = pt[3] - pt[1] 
            label = pt[4]
            if int(label) == 0 and w/h >3:
                continue
            if w*h > 400:
                tmp_bb.extend(pt)
        if len(tmp_bb) >0:
            bb_out = map(str,tmp_bb)
            bb_str = ','.join(list(bb_out))
            f_w.write('{},{}\n'.format(imgpath,bb_str))
    f_r.close()
    f_w.close()


if __name__=='__main__':
    #base_dir = "/data/VOC/VOCdevkit/VOC2012"
    #file_in = "/data/VOC/VOCdevkit/VOC2012/ImageSets/Main/trainval.txt"
    #file_out = "train_voc.txt"
    args = parms()
    base_dir = args.VOC_dir
    file_in = args.anno_file
    file_out = args.out_file
    dir_path = args.xml_dir
    file2_in = args.file2_in
    cmd = args.cmd_type
    #cmd = cmd.strip()
    #print(base_dir)
    #print(cmd)
    if cmd in ['readvoc']:
        convert_voc(base_dir,file_in,file_out,args.save_name)
    elif cmd == 'getdir':
        get_dir_cnt(dir_path,file_out)
    elif cmd == 'merge':
        merge_2file(file_in,file2_in,file_out)
    elif cmd == 'filter':
        filter_area(file_in,file_out)
    else:
        print('please input right cmd')