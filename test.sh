export CUDA_VISIBLE_DEVICES=2
cd demo/
python demo.py --config-file /mnt/SSD7/yuwei-hdd3/selected/HW3/detectron2/tools/exp_final/X_101_32x8d_FPN_3x_pretrained_deform_ratio/config.yaml \
  --input /mnt/SSD7/yuwei-hdd3/selected/HW3/hw3-data-release/test_release \
  --output ../results/deform_ratio_60k\
  --opts MODEL.WEIGHTS /mnt/SSD7/yuwei-hdd3/selected/HW3/detectron2/tools/exp_final/X_101_32x8d_FPN_3x_pretrained_deform_ratio/model_0059999.pth


  
