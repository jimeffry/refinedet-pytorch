#CUDA_VISIBLE_DEVICES=0 python train_refinedet.py --save_folder /wdc/LXY.data/models/pytorch --input_size 320 --dataset_root /wdc/LXY.data/VOC/
python train_refinedet.py --dataset cocovoc --cuda False --model_dir /data/models --save_weight_period 2 --gamma 10 --arm_gamma 10 --start_iter 0 --batch_size 2