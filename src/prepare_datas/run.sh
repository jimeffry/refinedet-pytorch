#!/usr/bin/bash
#******************************COCO
#python voc_process.py --VOC-dir /data/VOC/VOCdevkit/VOC2012 --anno-file voc2012anno.txt --out-file train_voc.txt
#python voc_process.py --xml-dir /data/VOC/VOCdevkit/VOC2012/Annotations  --out-file voc2012anno.txt
#python coco_process.py --image_dir /data/COCO/train2017 --annotation_path /data/COCO/annotations/instances_train2017.json --output_dir ../../datas --out_file coco2017train_new8.txt
#python coco_process.py --image_dir /data/COCO/val2017 --annotation_path /data/COCO/annotations/instances_val2017.json --output_dir ../../datas --out_file coco2017val_new8.txt
#**************************voc 07 12 process
#python voc_process.py --VOC-dir /data/VOC/VOCdevkit/VOC2007 --anno-file /data/VOC/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt --out-file ../../datas/VOC/trainval_voc07_5.txt --save-name trainval_record_voc07_5 --cmd-type readvoc
#python voc_process.py --VOC-dir=/data/VOC/VOCdevkit/VOC2012 --anno-file /data/VOC/VOCdevkit/VOC2012/ImageSets/Main/trainval.txt --out-file ../../datas/VOC/trainval_voc12_5.txt --save-name trainval_record_voc12_5 --cmd-type readvoc
#python voc_process.py --VOC-dir /data/VOC/VOCdevkit/VOC2007 --anno-file /data/VOC/VOCdevkit/VOC2007/ImageSets/Main/test.txt --out-file ../../datas/VOC/test_voc07.txt --save-name test_record --cmd-type readvoc

#*************************merge
#python voc_process.py --anno-file ../../datas/VOC/trainval_voc07_5.txt --file2-in ../../datas/VOC/trainval_voc12_5.txt --out-file ../../datas/VOC/trainval_voc0712_5.txt --cmd-type merge
#python voc_process.py --anno-file ../../datas/COCO/coco2017train.txt --file2-in ../../datas/COCO/coco2017val.txt --out-file ../../datas/COCO/coco2017.txt --cmd-type merge
python voc_process.py --anno-file ../../datas/COCO/coco2017train_new8.txt --file2-in ../../datas/COCO/coco2017val_new8.txt --out-file ../../datas/COCO/coco2017_8.txt --cmd-type merge

#python convert_to_pickle.py  --voc-dir /data/VOC/VOCdevkit/ --coco-dir /data/COCO/ --save-dir /data/train_record/ --voc-anno ../../datas/VOC/train_voc.txt --coco-anno ../../datas/COCO/coco2017.txt --out-file voc_coco.pkl --record-file ../../datas/train_all_record.txt
#python voc_process.py --anno-file ../../datas/COCO/coco2017m.txt --out-file ../../datas/COCO/coco2017_5.txt   --cmd-type filter