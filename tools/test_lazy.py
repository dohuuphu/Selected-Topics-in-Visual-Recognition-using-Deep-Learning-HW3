from detectron2.model_zoo import get_config
from detectron2.config import LazyConfig
print(LazyConfig.to_py(get_config("/mnt/SSD7/yuwei-hdd3/selected/HW3/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.py")))
