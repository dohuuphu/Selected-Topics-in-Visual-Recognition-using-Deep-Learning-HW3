import os
import json
import glob
import cv2
import numpy as np
import skimage.io as sio
from sklearn.model_selection import train_test_split

def generate_coco_json(folders, root_dir, output_json, start_image_id=0, start_annotation_id=0):
    coco_output = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    image_id = start_image_id
    annotation_id = start_annotation_id
    category_ids_set = set()

    for folder in folders:
        folder_path = os.path.join(root_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        img_path = os.path.join(folder_path, "image.tif")
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: cannot read image {img_path}, skipping...")
            continue
        height, width = img.shape[:2]

        coco_output["images"].append({
            "file_name": os.path.relpath(img_path, root_dir),
            "height": height,
            "width": width,
            "id": image_id
        })

        mask_paths = glob.glob(os.path.join(folder_path, "class*.tif"))
        for mask_path in mask_paths:
            filename = os.path.basename(mask_path)
            category_id = int(filename.replace("class", "").replace(".tif", ""))
            category_ids_set.add(category_id)

            mask = sio.imread(mask_path)
            if mask is None:
                print(f"Warning: cannot read mask {mask_path}, skipping...")
                continue
            if len(mask.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

            # Lấy tất cả instance ID khác 0
            instance_ids = np.unique(mask)
            instance_ids = instance_ids[instance_ids != 0]

            for inst_id in instance_ids:
                instance_mask = (mask == inst_id).astype(np.uint8)
                if instance_mask.sum() == 0:
                    continue

                polygons = mask_to_polygons(instance_mask)
                if len(polygons) == 0:
                    continue

                for poly in polygons:
                    coco_output["annotations"].append({
                        "segmentation": [poly],
                        "iscrowd": 0,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": get_bbox_from_poly(poly),
                        "area": cv2.contourArea(np.array(poly).reshape(-1, 2)),
                        "id": annotation_id
                    })
                    annotation_id += 1

        image_id += 1

    for cid in sorted(category_ids_set):
        coco_output["categories"].append({
            "id": cid,
            "name": f"class{cid}",
            "supercategory": "none"
        })

    with open(output_json, "w") as f:
        json.dump(coco_output, f, indent=2)

    return image_id, annotation_id


def mask_to_polygons(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        contour = contour.flatten().tolist()
        if len(contour) >= 6: 
            polygons.append(contour)
    return polygons


def get_bbox_from_poly(poly):
    poly = np.array(poly).reshape(-1, 2)
    x_min = np.min(poly[:, 0])
    y_min = np.min(poly[:, 1])
    x_max = np.max(poly[:, 0])
    y_max = np.max(poly[:, 1])
    return [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]


if __name__ == "__main__":
    root_dir = "../hw3-data-release/train"
    all_folders = sorted([f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))])

    train_folders, val_folders = train_test_split(all_folders, test_size=0.0, random_state=42)
# 
    print(f"Train: {len(train_folders)} images, Val: {len(val_folders)} images")

    last_img_id, last_ann_id = generate_coco_json(all_folders, root_dir, "train.json")
    generate_coco_json(val_folders, root_dir, "val.json", start_image_id=last_img_id, start_annotation_id=last_ann_id)
