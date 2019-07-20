#!/usr/bin/bash
########################run demo  test2017/000000000619.jpg  000000550601  000000182279
#python img_demo.py --img_path ../../datas/VOC/test_voc07.txt  --load_num 80000 --img_dir /data/VOC/VOCdevkit/VOC2007/JPEGImages --save_dir /data/test/voc_org9 
#python img_demo.py --img_path /data/COCO/val2017/000000002006.jpg --load_num 272
python img_demo.py --img_path /data/VOC/VOCdevkit/VOC2007/JPEGImages/000693.jpg --load_num 172 #144000
#python img_demo.py --img_path /data/test --load_num 80000 #130000
#python img_demo.py --img_path ../../datas/COCO/coco2017m.txt  --load_num 208000 --img_dir /data/COCO --save_dir /data/test/voc_org9
#python img_demo.py --img_path /data/new_way.avi --load_num 208000