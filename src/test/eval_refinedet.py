"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""
import sys
import os
import time
import argparse
import numpy as np
from  tqdm import tqdm
import pickle
import cv2
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

#from data import VOC_ROOT, VOCAnnotationTransform, VOCDetection, BaseTransform
#from data import VOC_CLASSES as labelmap
sys.path.append(os.path.join(os.path.dirname(__file__),'../utils'))
from detection_refinedet import Detect_RefineDet
sys.path.append(os.path.join(os.path.dirname(__file__),'../networks'))
from refinedet import build_refinedet
#from refinedet_train_test import build_refinedet
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfgs
sys.path.append(os.path.join(os.path.dirname(__file__),'../prepare_datas'))
from read_imgs import ReadDataset

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Evaluation')
parser.add_argument('--trained_model',
                    default='weights/ssd300_mAP_77.43_v2.pth', type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--voc_root',type=str, default=None,
                    help='Location of VOC root directory')
parser.add_argument('--cleanup', default=True, type=str2bool,
                    help='Cleanup and remove results files following eval')
parser.add_argument('--input_size', default='320', choices=['320', '512'],
                    type=str, help='RefineDet320 or RefineDet512')

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't using \
              CUDA.  Run with --cuda for optimal eval speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

annopath = os.path.join(args.voc_root, 'VOC2007', 'Annotations', '%s.xml')
imgpath = os.path.join(args.voc_root, 'VOC2007', 'JPEGImages', '%s.jpg')
imgsetpath = os.path.join(args.voc_root, 'VOC2007', 'ImageSets',
                          'Main', '{:s}.txt')
YEAR = '2007'
devkit_path = os.path.join(args.voc_root , 'VOC' + YEAR)
#save_dir = args.save_folder
dataset_mean = (104, 117, 123)
set_type = 'voctest'
labelmap = cfgs.VOCDataNames

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
                              int(bbox.find('ymin').text) - 1,
                              int(bbox.find('xmax').text) - 1,
                              int(bbox.find('ymax').text) - 1]
        objects.append(obj_struct)

    return objects


def get_output_dir(name, phase):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.
    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    filedir = os.path.join(name, phase)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    return filedir


def get_voc_results_file_template(image_set, cls_name,save_dir):
    # VOCdevkit/VOC2007/results/det_test_aeroplane.txt
    filename = 'det_' + image_set + '_%s.txt' % (cls_name)
    filedir = os.path.join(save_dir, 'results')
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    path = os.path.join(filedir, filename)
    return path


def write_voc_results_file(all_boxes, dataset,save_dir):
    for cls_ind, cls_name in enumerate(labelmap):
        print('Writing {:s} VOC results file'.format(cls_name))
        filename = get_voc_results_file_template(set_type, cls_name,save_dir)
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(dataset.ids):
                dets = all_boxes[cls_ind+1][im_ind]
                if dets == []:
                    continue
                # the VOCdevkit expects 1-based indices
                for k in range(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index[1], dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1))


def do_python_eval(output_dir='output', use_07=True):
    cachedir = os.path.join(output_dir, 'annotations_cache')
    aps = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = use_07
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for i, cls_name in enumerate(labelmap):
        filename = get_voc_results_file_template(set_type, cls_name,output_dir)
        rec, prec, ap, pos_num = voc_eval(filename, annopath, imgsetpath.format(set_type), cls_name, cachedir,
           ovthresh=0.5, use_07_metric=use_07_metric)
        aps += [ap]
        print('AP for {} = {:.4f}'.format(cls_name, ap))
        with open(os.path.join(output_dir, cls_name + '_pr.txt'), 'w') as f:
            #pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
            for tmp_id in range(len(rec)):
                f.write("rec: {:.3f},prec: {:.3f}\n".format(rec[tmp_id],prec[tmp_id]))
            f.write('ap: {:.3f}, pos_num: {}'.format(ap,pos_num))
    print('Mean AP = {:.4f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('Results:')
    for ap in aps:
        print('{:.3f}'.format(ap))
    print('{:.3f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('')
    print('--------------------------------------------------------------')
    print('Results computed with the **unofficial** Python eval code.')
    print('Results should be very close to the official MATLAB eval code.')
    print('--------------------------------------------------------------')


def voc_ap(rec, prec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:True).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        rec_num = 0
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
                rec_num +=1
            ap = ap + p 
        ap = ap / float(rec_num)
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=True):
    """rec, prec, ap = voc_eval(detpath,
                           annopath,
                           imagesetfile,
                           classname,
                           [ovthresh],
                           [use_07_metric])
Top level function that does the PASCAL VOC evaluation.
detpath: Path to detections
   detpath.format(classname) should produce the detection results file.
annopath: Path to annotations
   annopath.format(imagename) should be the xml annotations file.
imagesetfile: Text file containing the list of images, one image per line.
classname: Category name (duh)
cachedir: Directory for caching the annotations
[ovthresh]: Overlap threshold (default = 0.5)
[use_07_metric]: Whether to use VOC07's 11 point AP computation
   (default True)
"""
# assumes detections are in detpath.format(classname)
# assumes annotations are in annopath.format(imagename)
# assumes imagesetfile is a text file with each line an image name
# cachedir caches the annotations in a pickle file
# first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath % (imagename))
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                   i + 1, len(imagenames)))
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(1 - difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    #detfile = detpath.format(classname)
    detfile = detpath
    with open(detfile, 'r') as f:
        detection_cnts = f.readlines()
    image_ids = []
    confidence = []
    boxes = []
    if len(detection_cnts) >=1:
        for tmp_cnt in detection_cnts:
            tmp_splits = tmp_cnt.strip().split()
            image_ids.append(tmp_splits[0])
            confidence.append(float(tmp_splits[1]))
            tmp_box = map(float,tmp_splits[2:])
            boxes.append(list(tmp_box))
        '''
        splitlines = [x.strip().split() for x in detection_cnts]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])
        '''
        confidence = np.array(confidence)
        boxes = np.array(boxes)
        # sort by confidence 
        # could select top k detections boxes
        sorted_ind = np.argsort(confidence)[::-1]
        sorted_scores = np.sort(confidence)[::-1]
        boxes = boxes[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        detect_nums = len(image_ids)
        tp = np.zeros(detect_nums)
        fp = np.zeros(detect_nums)
        for idx in range(detect_nums):
            annotations_gt = class_recs[image_ids[idx]]
            bb = boxes[idx, :].astype(float)
            ovmax = -np.inf
            bbox_gt = annotations_gt['bbox'].astype(float)
            if bbox_gt.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(bbox_gt[:, 0], bb[0])
                iymin = np.maximum(bbox_gt[:, 1], bb[1])
                ixmax = np.minimum(bbox_gt[:, 2], bb[2])
                iymax = np.minimum(bbox_gt[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)
                inters = iw * ih
                uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (bbox_gt[:, 2] - bbox_gt[:, 0]) *
                       (bbox_gt[:, 3] - bbox_gt[:, 1]) - inters)
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not annotations_gt['difficult'][jmax]:
                    if not annotations_gt['det'][jmax]:
                        tp[idx] = 1.
                        annotations_gt['det'][jmax] = 1
                    else:
                        fp[idx] = 1.
            else:
                fp[idx] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
    else:
        rec = -1.
        prec = -1.
        ap = -1.

    return rec, prec, ap,npos


def test_net(save_folder, net,detect_model, cuda, dataset, top_k,
             im_size=320, thresh=0.05):
    num_images = len(dataset.shulf_num)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap)+1)]

    # timers
    #_t = {'im_detect': Timer(), 'misc': Timer()}
    output_dir = get_output_dir(save_folder, set_type)
    det_file = os.path.join(output_dir, 'detections.pkl')
    if os.path.exists(det_file):
        f = open(det_file,'rb')
        all_boxes = pickle.load(f)
    else:
        for i in tqdm(range(num_images)):
            im, gt, h, w = dataset.pull_item(i)
            x = Variable(im.unsqueeze(0))
            if args.cuda:
                x = x.cuda()
            #_t['im_detect'].tic()
            #detections = net(x).data
            arm_loc_data, arm_conf_data, odm_loc_data, odm_conf_data, priors,_ = net(x)
            detections = detect_model(arm_loc_data, arm_conf_data, odm_loc_data, odm_conf_data, priors)
            detections = detections.data
            #detect_time = _t['im_detect'].toc(average=False)

            # skip j = 0, because it's the background class
            for j in range(1, detections.size(1)):
                dets = detections[0, j, :]
                mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t() #lxy score threshold
                dets = torch.masked_select(dets, mask).view(-1, 5)
                if dets.size(0) == 0:
                    continue
                boxes = dets[:, 1:]
                boxes[:, 0] *= w
                boxes[:, 2] *= w
                boxes[:, 1] *= h
                boxes[:, 3] *= h
                scores = dets[:, 0].cpu().numpy()
                cls_dets = np.hstack((boxes.cpu().numpy(),
                                    scores[:, np.newaxis])).astype(np.float32,
                                                                    copy=False)
                all_boxes[j][i] = cls_dets

            #print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1,
             #                                           num_images, detect_time))

        with open(det_file, 'wb') as f:
            pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
        write_voc_results_file(all_boxes, dataset,output_dir)
    print('Evaluating detections')
    #f = open(det_file,'rb')
    #all_boxes = pickle.load(f)
    evaluate_detections(all_boxes, output_dir, dataset)


def evaluate_detections(box_list, output_dir, dataset):
    write_voc_results_file(box_list, dataset,output_dir)
    do_python_eval(output_dir)


if __name__ == '__main__':
    # load net
    
    num_classes = len(cfgs.VOCDataNames) + 1  # +1 for background
    net = build_refinedet()
    Detector = Detect_RefineDet()
    if args.cuda :
        net.load_state_dict(torch.load(args.trained_model))
    else:
        net.load_state_dict(torch.load(args.trained_model,map_location='cpu'))
    net.eval()
    print('Finished loading model!')
    # load data
    #dataset = VOCDetection(args.voc_root, [('2007', set_type)],
     #                      BaseTransform(int(args.input_size), dataset_mean),
      #                     VOCAnnotationTransform())
    dataset = ReadDataset()
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(args.save_folder, net,Detector, args.cuda, dataset, args.top_k, int(args.input_size),
             thresh=args.confidence_threshold)
    