

cd tools/
#  CUDA_VISIBLE_DEVICES=1 ./train_net.py --num-gpus 1 --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml

#  CUDA_VISIBLE_DEVICES=2 ./train_net.py --num-gpus 1 --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml

#  CUDA_VISIBLE_DEVICES=3 ./train_net.py --num-gpus 1 --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml --resume

CUDA_VISIBLE_DEVICES=4 ./train_net.py --num-gpus 1 --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x_pretrained.yaml
CUDA_VISIBLE_DEVICES=4 ./train_net.py --num-gpus 1 --config-file ../configs/hw3/X_101_32x8d_FPN_3x_AUG.yaml

CUDA_VISIBLE_DEVICES=4 ./train_net.py --num-gpus 1 --config-file ../configs/hw3/X_101_32x8d_FPN_3x_AUG_res.yaml
CUDA_VISIBLE_DEVICES=5 ./train_net.py --num-gpus 1 --config-file ../configs/hw3/X_101_32x8d_FPN_3x_AUG_res_CE.yaml

