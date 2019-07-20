#author: lxy
#time: 14:30 2019.7.6
#tool: python3
#version: 0.1
#project: pedestrian detection
#modify:
#####################################################
import sys
import os
import numpy as np 
from matplotlib import pyplot as plt 
import argparse

def parms():
    parser = argparse.ArgumentParser(description='refinedet traing log')
    parser.add_argument('--file_in', default=None,
                        type=str, help='log file')
    parser.add_argument('--data_name', default='voc', type=str,
                        help='traing data name')
    parser.add_argument('--loss_name',type=str,default='total',help='loss')
    parser.add_argument('--load_num',type=int,default=0,help='load model num')
    return parser.parse_args()

def read_data(file_in,name):
    '''
    file_in: log file 
        epoch || loss || lr || speed
    '''
    f_r = open(file_in,'r')
    file_cnts = f_r.readlines()
    epoch_datas = []
    arm_l = []
    arm_c =[]
    odm_l = []
    odm_c = []
    total_loss = []
    i = 0
    for tmp in file_cnts:
        i+=1
        tmp_splits = tmp.strip().split('||')
        if len(tmp_splits) <2:
            continue
        splits0 = tmp_splits[0].strip().split()
        #print(splits0)
        splits1 = tmp_splits[1].strip().split()
        #print(splits1)
        if name == 'org':
            epoch_datas.append(int(splits0[-1]))
            total_loss.append(float(splits1[-1].split(':')[-1]))
        else:
            #epoch_datas.append(int(splits0[0][6:]))
            epoch_datas.append(i)
            total_loss.append(float(splits1[-1]))
        arm_l.append(float(splits1[2]))
        arm_c.append(float(splits1[5]))
        odm_l.append(float(splits1[8]))
        odm_c.append(float(splits1[11]))
        #if i==1:
         #   break
    loss_datas = [arm_l,arm_c,odm_l,odm_c,total_loss]
    return epoch_datas,loss_datas

def plot_lines(txt_path,name,key_name):
    ax_data,total_data = read_data(txt_path,key_name)
    fig = plt.figure(num=0,figsize=(20,10))
    plt.plot(ax_data,total_data[-1],label='total' )
    plt.plot(ax_data,total_data[0],label='arm_loc')
    plt.plot(ax_data,total_data[1],label='arm_conf')
    plt.plot(ax_data,total_data[2],label='odm_loc')
    plt.plot(ax_data,total_data[3],label='odm_conf')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title('%s_training' % name)
    plt.grid(True)
    leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.2)
    #leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)
    plt.savefig("../../logs/%s.png" % name ,format='png')
    plt.show()

def plot_pr(file_in,name):
    '''
    file_in: file record rec and prec 
    '''
    f_r = open(file_in,'r')
    f_cnts = f_r.readlines()
    rec_dict = dict()
    total = len(f_cnts)
    for idx,tmp in enumerate(f_cnts):
        if idx == total-1:
            continue
        tmp_splits = tmp.strip().split(',')
        rec = tmp_splits[0].split()[-1]
        prec = float(tmp_splits[1].split()[-1])
        value_ = rec_dict.setdefault(rec,0)
        if value_ < prec:
            rec_dict[rec] = prec
    f_r.close()
    # plot
    rec_keys = rec_dict.keys()
    prec_data = []
    for tmp_key in rec_keys:
        prec_data.append(rec_dict[tmp_key])
    rec_data = list(map(float,rec_keys))
    ax_data = rec_data
    ay_data = prec_data
    plt.plot(ax_data,ay_data,label='ROC')
    plt.ylabel('precision')
    plt.xlabel('recall')
    plt.title('%s_roc' % name)
    plt.grid(True)
    leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.2)
    #leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)
    plt.savefig("../../logs/%s.png" % name ,format='png')
    plt.show()

def plot_roces(base_dir):
    '''
    '''
    name_list = ['person','bicycle','motorbike','car','bus','aeroplane','train','boat']
    for tmp in name_list:
        input_file = os.path.join(base_dir,tmp+'_pr.txt')
        plot_pr(input_file,tmp)
    

if __name__=='__main__':
    args = parms()
    file_in = args.file_in
    data_name = args.data_name
    loss_name = args.loss_name
    #plot_lines(file_in,data_name,loss_name)
    #plot_pr(file_in,data_name)
    plot_roces(file_in)