import torch
from torch.autograd import Function
from torch.nn.Function as F
from box_utils import decode, nms, center_size
from data import voc_refinedet as cfg


#class Detect_RefineDet(Function):
class Detect_RefineDet(object):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, num_classes, size, bkg_label, top_k, conf_thresh, nms_thresh, 
                objectness_thre, keep_top_k):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        self.keep_top_k = keep_top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.objectness_thre = objectness_thre
        self.variance = cfg[str(size)]['variance']
        if torch.cuda.is_available():
            self.variance = torch.tensor(self.variance,dtype=torch.float)
            self.variance.cuda()

    def forward(self, arm_loc_data, arm_conf_data, odm_loc_data, odm_conf_data, prior_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [num_priors,4]
        """
        loc_data = odm_loc_data
        conf_data = F.softmax(odm_conf_data,dim=2)
        arm_conf_data = F.softmax(arm_conf_data,dim=2)

        arm_object_conf = arm_conf_data.data[:, :, 1:]
        no_object_index = arm_object_conf <= self.objectness_thre
        conf_data[no_object_index.expand_as(conf_data)] = 0

        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        conf_preds = conf_data.view(num, num_priors,
                                   self.num_classes).transpose(2, 1)
        #conf_preds = conf_data.view(num,num_priors,self.num_classes)
        # Decode predictions into bboxes.
        if torch.cuda.is_available():
            prior_data.cuda()
        for i in range(num):
            default = decode(arm_loc_data[i], prior_data, self.variance)
            default = center_size(default)
            decoded_boxes = decode(loc_data[i], default, self.variance)
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()
            '''
            prior_conf_max,prior_conf_idx = conf_scores.max(1,keepdim=True)
            cls_mask = prior_conf_idx.gt(0)
            prior_conf_max = prior_conf_max[cls_mask]
            prior_conf_idx = prior_conf_idx[cls_mask]
            decoded_boxes = decoded_boxes[cls_mask]
            conf_mask = prior_conf_max.gt(self.conf_thresh)
            prior_conf_max = prior_conf_max[conf_mask]
            prior_conf_idx = prior_conf_idx[conf_mask]
            decoded_boxes = decoded_boxes[conf_mask]
            '''
            #print(decoded_boxes, conf_scores)
            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                #print(scores.dim())
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # idx of highest scoring and non-overlapping boxes per class
                #print(boxes, scores)
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                output[i, cl, :count] = \
                    torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]]), 1)
        #flt = output.contiguous().view(num, -1, 5)
        #_, idx = flt[:, :, 0].sort(1, descending=True)
        #_, rank = idx.sort(1)                                             ############????????
        #flt[(rank < self.keep_top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        #print('fit',output.size())
        return output
