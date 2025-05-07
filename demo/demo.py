# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm
import json
import zipfile

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import GenericMask
import pycocotools.mask as mask_util
import utils
# from vision.fair.detectron2.demo.predictor import VisualizationDemo
from predictor import VisualizationDemo
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog

# constants
WINDOW_NAME = "COCO detections"

# register_coco_instances("my_dataset_test", {}, "/mnt/SSD7/yuwei-hdd3/selected/HW3/detectron2/demo/coco_predictions.json", "/mnt/SSD7/yuwei-hdd3/selected/HW3/hw3-data-release/test_release")


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold

    cfg.DATASETS.TEST = ("my_dataset_test",)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


def main() -> None:
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)
    outputs = [] 

    # with open('/mnt/SSD7/yuwei-hdd3/selected/HW3/hw3-data-release/val.json', 'r') as f:
    #     input_data = json.load(f)

    with open('/mnt/SSD7/yuwei-hdd3/selected/HW3/hw3-data-release/test_image_name_to_ids.json', 'r') as f:
        input_data = json.load(f)

    # input_data = json.load('/mnt/SSD7/yuwei-hdd3/selected/HW3/hw3-data-release/test_image_name_to_ids.json')
    for idx, image_data  in enumerate(tqdm.tqdm(input_data, disable=not args.output)):
        # use PIL, to be consistent with evaluation
        image_entry = {
                "id": image_data["id"],
                "file_name": image_data["file_name"],
                "height": image_data["height"],
                "width": image_data["width"]
            }
        path = os.path.join(args.input[0],image_data["file_name"])
        img = read_image( path, format="BGR")
        start_time = time.time()
        predictions, visualized_output = demo.run_on_image(img)
        logger.info(
            "{}: {} in {:.2f}s".format(
                path,
                (
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished"
                ),
                time.time() - start_time,
            )
        )


        if not os.path.isdir(os.path.join(args.output,'imgs')):
            os.makedirs(os.path.join(args.output,'imgs'))
        out_filename = os.path.join(args.output,'imgs', os.path.basename(path))
        # out_filename = os.path.join(args.output,'imgs', path.split('/')[-2]+'.jpg') #val


        visualized_output.save(out_filename)


        if "instances" in predictions:
            instances = predictions["instances"].to("cpu")
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.numpy()
            classes = instances.pred_classes.numpy()

            has_mask = instances.has("pred_masks")
            if has_mask:
                masks = instances.pred_masks.numpy().astype(np.uint8)
                height, width = masks.shape[1:]

            for i in range(len(boxes)):
                box = boxes[i]
                score = scores[i]
                category_id = int(classes[i])+1

                coco_output = {
                    "image_id": int(image_data["id"]),
                    "bbox": [
                        float(box[0]),
                        float(box[1]),
                        float(box[2] - box[0]),
                        float(box[3] - box[1])
                    ],
                    "score": float(score),
                    "category_id": category_id,
                }

                if has_mask:
                    mask = masks[i]
                    rle = utils.encode_mask(mask) 
                    coco_output["segmentation"] = {
                        "size": [height, width],
                        "counts": rle["counts"]
                    }

                outputs.append(coco_output)

    # Save prediction to JSON file
    json_path = f"{args.output}/test-results.json"
    with open(json_path, "w") as f:
        json.dump(outputs, f)

    zip_path = f"{args.output}/{os.path.basename(args.output)}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(json_path, arcname="test-results.json") 
    

if __name__ == "__main__":
    main()  # pragma: no cover