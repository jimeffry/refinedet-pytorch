 #!/usr/bin/bash
 python eval_refinedet.py --trained_model /data/models/refinedet/RefineDet320_VOC_292000.pth  \
   --save_folder /data/VOC/VOCdevkit/VOC2007/output_anc10 --voc_root /data/VOC/VOCdevkit --cuda Fasle