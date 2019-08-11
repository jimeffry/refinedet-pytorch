#!/usr/bin/bash
#python histogram.py --file_in ../../logs/log_6.txt  --data_name cocovoc6 --loss_name org --cmd_type plot_trainlog
#python histogram.py --file_in ../../logs/2019-07-13-11-14-18.log  --data_name cocovoc_m2 --loss_name modify --cmd_type plot_trainlog
#**********plot roc ['person','bicycle','motorbike','car','bus','aeroplane','train','boat']
#python histogram.py --file_in /data/VOC/VOCdevkit/VOC2007/output_org9/voctest/  --cmd_type plot_pr
python histogram.py --file_in /data/VOC/VOCdevkit/VOC2007/m2det/ --cmd_type plot_pr
#python histogram.py --file_in ../../datas/VOC/trainval_record_voc12.txt --data_name voctrain --file2_in ../../datas/VOC/trainval_record_voc07.txt --cmd_type plot_data
#python histogram.py --file_in ../../datas/COCO/record_cocotrain.txt --data_name cocotrain  --cmd_type plot_data