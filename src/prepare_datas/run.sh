#!/usr/bin/bash
#python voc_process.py --VOC-dir /data/VOC/VOCdevkit/VOC2012 --anno-file voc2012anno.txt --out-file train_voc.txt
#python voc_process.py --xml-dir /data/VOC/VOCdevkit/VOC2012/Annotations  --out-file voc2012anno.txt
#python voc_process.py --VOC-dir /data/VOC/VOCdevkit/VOC2007 --anno-file /data/VOC/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt --out-file train_voc07.txt --cmd-type readvoc
#python voc_process.py --anno-file ../../datas/COCO/coco2017train.txt --file2-in ../../datas/COCO/coco2017val.txt --out-file ../../datas/COCO/coco2017.txt --cmd-type merge

python convert_to_pickle.py  --voc-dir /data/VOC/VOCdevkit/ --coco-dir /data/COCO/ --save-dir /data/train_record/ --voc-anno ../../datas/VOC/train_voc.txt --coco-anno ../../datas/COCO/coco2017.txt --out-file voc_coco.pkl --record-file ../../datas/train_all_record.txt