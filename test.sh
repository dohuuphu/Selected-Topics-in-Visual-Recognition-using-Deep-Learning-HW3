export CUDA_VISIBLE_DEVICES=2
cd demo/
python demo.py --config-file ../tools/exp/SyncBN/config.yaml \
  --input ../hw3-data-release/test_release \
  --output ../results/SyncBN\
  --opts MODEL.WEIGHTS ../tools/exp/SyncBN/model_0059999.pth


  
