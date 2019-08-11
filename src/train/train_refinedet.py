#author: lxy
#time: 14:30 2019.7.2
#tool: python3
#version: 0.1
#project: pedestrian detection
#modify:
#####################################################
import os
import sys
import time
import logging
import numpy as np
import argparse
#import visdom
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
sys.path.append(os.path.join(os.path.dirname(__file__),'../losses'))
from refinedet_multibox_loss import RefineDetMultiBoxLoss
sys.path.append(os.path.join(os.path.dirname(__file__),'../networks'))
#from refinedet import build_refinedet
from refinedet_resnet import build_refinedet
sys.path.append(os.path.join(os.path.dirname(__file__),'../prepare_datas'))
from read_pkle import detection_collate,ReadPkDataset
from read_imgs import ReadDataset
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfgs

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def args():
    parser = argparse.ArgumentParser(
        description='Single Shot MultiBox Detector Training With Pytorch')
    parser.add_argument('--dataset', default='VOC',
                        type=str, help='VOC or COCO')
    parser.add_argument('--gpu', default='0', 
                        type=str, help='gpu')
    parser.add_argument('--log_dir',type=str, default='../../logs',
                        help='log dir')
    parser.add_argument('--basenet', default='res101.pth',
                        help='Pretrained base model')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size for training')
    parser.add_argument('--load_num', default=None, type=str,
                        help='Checkpoint state_dict file to resume training from')
    parser.add_argument('--start_iter', default=0, type=int,
                        help='Resume training at this iter')
    parser.add_argument('--num_workers', default=1, type=int,
                        help='Number of workers used in dataloading')
    parser.add_argument('--cuda', default=True, type=str2bool,
                        help='Use CUDA to train model')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        help='initial learning rate')
    parser.add_argument('--odm_gamma', default=10, type=float,
                        help='Gamma  loss')
    parser.add_argument('--arm_gamma', default=10, type=float,
                        help='Gamma  loss')
    parser.add_argument('--visdom', default=False, type=str2bool,
                        help='Use visdom for loss visualization')
    parser.add_argument('--model_dir', default='../../models',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--save_weight_period', default=5,type=int,
                        help='save weights')
    args = parser.parse_args()
    return args

def set_config(args):
    if torch.cuda.is_available():
        if args.cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        if not args.cuda:
            print("WARNING: It looks like you have a CUDA device, but aren't " +
                "using CUDA.\nRun with --cuda for optimal training speed.")
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

def train(args):
    #*********************************************load args
    log_dir = args.log_dir
    model_dir = args.model_dir
    dataname = args.dataset
    load_num = args.load_num
    batch_size = args.batch_size
    save_weight_period = args.save_weight_period
    gpu_list = [int(i) for i in args.gpu.split(',')]
    gpu_num = len(gpu_list)
    #********************************************creat logging
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = logging.getLogger()
    log_name = time.strftime('%F-%T',time.localtime()).replace(':','-')+'.log'
    log_path = os.path.join(log_dir,log_name)
    hdlr = logging.FileHandler(log_path)
    logger.addHandler(hdlr)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)
    logger.info("train gpu:{}".format(gpu_list))
    #*************************************************load data io
    train_dataset = ReadDataset()
    #************************************************* build net
    refinedet_net = build_refinedet()
    net = refinedet_net
    #logger.info("**********> net struct:{} ".format(refinedet_net))
    if args.cuda:
        net = torch.nn.DataParallel(refinedet_net,gpu_list)
        torch.backends.cudnn.benchmark = False
    #************************************************** build summary
    if args.visdom:
        viz = visdom.Visdom()
        vis_title = 'RefineDet ' + dataname
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)
    #*****************************************************init network
    model_path = os.path.join(model_dir,dataname)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_path = os.path.join(model_path,cfgs.ModelPrefix)
    if load_num is not None:
        load_path = "%s_%s.pth" %(model_path,load_num)
        logger.info('Resuming training, loading {}...'.format(load_path))
        refinedet_net.load_weights(load_path)
    else:
        base_path = os.path.join(model_dir, args.basenet)
        base_weights = torch.load(base_path)
        logger.info('Loading base network weights...')
        refinedet_net.backone.load_state_dict(base_weights)
    #**********************************************************************build optimizer
    if args.cuda:
        net = net.cuda()
    if load_num is None:
        logger.info('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        refinedet_net.Arm_list.apply(weights_init)
        refinedet_net.Odm_list.apply(weights_init)
        refinedet_net.FPN.apply(weights_init)
        #refinedet_net.res6.apply(weights_init)
    optimizer = optim.SGD([{'params': net.parameters(), 'initial_lr': args.lr}], lr=args.lr, 
                            momentum=cfgs.Momentum,weight_decay=cfgs.Weight_decay)
    #optimizer = optim.Adam([{'params': net.parameters(), 'initial_lr': args.lr}], lr=args.lr, 
     #                       weight_decay=cfgs.Weight_decay)
    arm_criterion = RefineDetMultiBoxLoss(2, 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda)
    odm_criterion = RefineDetMultiBoxLoss(cfgs.ClsNum, 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda, use_ARM=True)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfgs.lr_steps, 
                            gamma=cfgs.lr_gamma, last_epoch=args.start_iter)
    #***********************************************************************build trainer
    net.train()
    # loss counters
    odm_gamma = torch.tensor(args.odm_gamma,dtype=torch.float)
    arm_gamma = torch.tensor(args.arm_gamma,dtype=torch.float)
    arm_loc_loss = 0
    arm_conf_loss = 0
    odm_loc_loss = 0
    odm_conf_loss = 0
    logger.info('Loading the dataset...')
    batch_num = int(np.ceil(float(train_dataset.__len__()) / float(batch_size)))
    logger.info('Training RefineDet on:{}'.format(dataname))
    logger.info('Using the specified args:{}'.format(args))
    #data_loader = torch.utils.data.DataLoader(train_dataset, batch_size,
     #                             num_workers=args.num_workers,
      #                            shuffle=True, collate_fn=detection_collate,
       #                           pin_memory=True)
    # create batch iterator
    #batch_iterator = iter(data_loader)
    for epoch_idx in range(args.start_iter, cfgs.epoch_num):
        for step_idx in range(batch_num):
            if args.visdom and step_idx != 0 and (step_idx % cfgs.Smry_iter == 0):
                update_vis_plot(step_idx, arm_loc_loss, arm_conf_loss, epoch_plot, None,
                            'append', cfgs.Smry_iter)
                arm_loc_loss = 0
                arm_conf_loss = 0
                odm_loc_loss = 0
                odm_conf_loss = 0
            # load train data
            try:
                #images, targets = next(batch_iterator)
                images, targets = train_dataset.get_batch(batch_size)
            except StopIteration:
                #train_dataset.close_pkl()
                #train_dataset.load_pkl()
                images, targets = train_dataset.get_batch(batch_size)
            if args.cuda:
                images = images.cuda()
                targets = [ann.cuda() for ann in targets]
                gamma.cuda()
                arm_gamma.cuda()
            else:
                images = images
                targets = [ann for ann in targets]
            # forward
            t0 = time.time()
            out = net(images)
            # backprop
            optimizer.zero_grad()
            arm_loss_l, arm_loss_c = arm_criterion(out, targets)
            odm_loss_l, odm_loss_c = odm_criterion(out, targets)
            #input()
            arm_loss = arm_loss_l * arm_gamma + arm_loss_c
            odm_loss = odm_loss_l * odm_gamma + odm_loss_c
            loss = arm_loss + odm_loss
            if bool(loss == 0):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)
            optimizer.step()
            t1 = time.time()
            if args.visdom:
                arm_loc_loss += arm_loss_l.item()
                arm_conf_loss += arm_loss_c.item()
                odm_loc_loss += odm_loss_l.item()
                odm_conf_loss += odm_loss_c.item()
            #adject lr
            scheduler.step()
            if step_idx % cfgs.Show_train_info == 0:
                training_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                logger.info('{}'.format(training_time))
                logger.info('epoch '+repr(epoch_idx)+' iter ' + repr(step_idx) + ' || ARM_L Loss: %.4f ARM_C Loss: %.4f ODM_L Loss: %.4f ODM_C Loss: %.4f Total_Loss: %.4f || lr: %.6f || speed: %.2f per Sec.' \
                % (arm_loss_l.item(), arm_loss_c.item(), odm_loss_l.item(), odm_loss_c.item(),loss.item(),optimizer.param_groups[0]['lr'],batch_size/(t1 - t0)))
            #if args.visdom:
             #   update_vis_plot(iteration, arm_loss_l.data[0], arm_loss_c.data[0],
              #                  iter_plot, epoch_plot, 'append')
        if epoch_idx % save_weight_period ==0 and epoch_idx > 0 :
            torch.save(refinedet_net.state_dict(),"{}_{}.pth".format(model_path,epoch_idx))
            logger.info(' *********************************** save weightes ')


def adjust_learning_rate(optimizer, lr,gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(iteration, loc, conf, window1, window2, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 3)).cpu(),
            Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )


if __name__ == '__main__':
    parms = args()
    set_config(parms)
    train(parms)
