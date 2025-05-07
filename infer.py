from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

# 1. Load config
cfg = get_cfg()
cfg.merge_from_file('/mnt/SSD7/yuwei-hdd3/selected/HW3/detectron2/tools/exp/X_101_32x8d_FPN_3x./config.yaml')
cfg.MODEL.WEIGHTS = '/path/to/your/model_final.pth'  # <- chỉ định file model weight nếu cần

# 2. Build model
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)  # load model từ checkpoint

# 3. Prepare evaluator & data loader
output_dir = 'results/testtt'
evaluator = COCOEvaluator("test", output_dir=output_dir)
val_loader = build_detection_test_loader(cfg, "test")

# 4. Run evaluation
inference_on_dataset(trainer.model, val_loader, evaluator)
