#!/usr/bin/env python3
"""
Merge a YOLO-format dataset (dataset_A) with a Pascal VOC-style archive (dataset_B_root)
which contains subfolders like Czech/ India/ Japan/ each with images/ and annotations/xmls/.

Outputs a merged YOLO dataset in `output_dir`:
output_dir/
  images/
  labels/
  train.txt
  val.txt
  test.txt
  data.yaml
"""

import os
import shutil
import xml.etree.ElementTree as ET
import random

# ---------- helpers ----------
def parse_pbtxt(pbtxt_path):
    names = []
    if not os.path.isfile(pbtxt_path):
        return names
    with open(pbtxt_path, "r", encoding="utf-8") as f:
        text = f.read()
    for block in text.split("item"):
        if "name" in block:
            idx = block.find("name")
            rest = block[idx:]
            s = rest.split("name:")[-1].strip()
            if "'" in s or '"' in s:
                name = s.split("'")[1] if "'" in s else s.split('"')[1]
                names.append(name)
    return names

def voc_to_yolo(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    if size is None:
        return []
    width = float(size.find("width").text)
    height = float(size.find("height").text)
    objs = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        bnd = obj.find("bndbox")
        xmin = float(bnd.find("xmin").text)
        ymin = float(bnd.find("ymin").text)
        xmax = float(bnd.find("xmax").text)
        ymax = float(bnd.find("ymax").text)
        x_center = (xmin + xmax) / 2.0
        y_center = (ymin + ymax) / 2.0
        w = xmax - xmin
        h = ymax - ymin
        x_center /= width
        y_center /= height
        w /= width
        h /= height
        objs.append((name, x_center, y_center, w, h))
    return objs

# ---------- main merging function ----------
def merge_datasets(yolo_A_path, voc_root, output_dir, pbtxt_path=None, classes_override=None,
                   train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    out_images = os.path.join(output_dir, "images")
    out_labels = os.path.join(output_dir, "labels")
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_labels, exist_ok=True)

    # 1) get class list
    classes = []
    if classes_override:
        classes = classes_override
    elif pbtxt_path and os.path.isfile(pbtxt_path):
        classes = parse_pbtxt(pbtxt_path)
    if not classes:
        classes = ["pothole"]

    class_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"Using classes: {classes}")

    all_image_paths = []

    # 2) copy YOLO A dataset (assume structure: yolo_A_path/images & yolo_A_path/labels)
    if yolo_A_path and os.path.isdir(yolo_A_path):
        imgs_dir = os.path.join(yolo_A_path, "images")
        labels_dir = os.path.join(yolo_A_path, "labels")
        if os.path.isdir(imgs_dir):
            for fn in os.listdir(imgs_dir):
                src_img = os.path.join(imgs_dir, fn)
                if not os.path.isfile(src_img):
                    continue
                new_name = f"A_{fn}"
                dst_img = os.path.join(out_images, new_name)
                shutil.copy2(src_img, dst_img)
                base, _ = os.path.splitext(fn)
                lbl_name = base + ".txt"
                src_lbl = os.path.join(labels_dir, lbl_name)
                dst_lbl = os.path.join(out_labels, os.path.splitext(new_name)[0] + ".txt")
                if os.path.isfile(src_lbl):
                    shutil.copy2(src_lbl, dst_lbl)
                else:
                    open(dst_lbl, "w").close()
                all_image_paths.append(dst_img)
        else:
            print("Warning: YOLO A images folder not found at", imgs_dir)
    else:
        print("Warning: No YOLO A dataset provided or path invalid.")

    # 3) convert VOC datasets (search subfolders)
    if os.path.isdir(voc_root):
        for sub in sorted(os.listdir(voc_root)):
            subdir = os.path.join(voc_root, sub)
            if not os.path.isdir(subdir):
                continue
            images_dir = os.path.join(subdir, "images")
            ann_dir = os.path.join(subdir, "annotations")
            xml_dir = None
            if os.path.isdir(ann_dir):
                if os.path.isdir(os.path.join(ann_dir, "xmls")):
                    xml_dir = os.path.join(ann_dir, "xmls")
                else:
                    xml_dir = ann_dir
            if not os.path.isdir(images_dir):
                for d in os.listdir(subdir):
                    if d.lower().startswith("image") and os.path.isdir(os.path.join(subdir, d)):
                        images_dir = os.path.join(subdir, d)
                        break
            if not xml_dir:
                for root, _, files in os.walk(subdir):
                    if any(f.endswith(".xml") for f in files):
                        xml_dir = root
                        break

            if not os.path.isdir(images_dir) or not os.path.isdir(xml_dir):
                print(f"Skipping {sub}: images or xmls not found. images_dir={images_dir}, xml_dir={xml_dir}")
                continue

            for img_fn in os.listdir(images_dir):
                if not img_fn.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                src_img = os.path.join(images_dir, img_fn)
                prefix = f"{sub}_"
                new_img_name = prefix + img_fn
                dst_img = os.path.join(out_images, new_img_name)
                shutil.copy2(src_img, dst_img)

                base = os.path.splitext(img_fn)[0]
                xml_name = base + ".xml"
                xml_path = os.path.join(xml_dir, xml_name)
                if not os.path.isfile(xml_path):
                    for f in os.listdir(xml_dir):
                        if os.path.splitext(f)[0].lower() == base.lower() and f.lower().endswith(".xml"):
                            xml_path = os.path.join(xml_dir, f)
                            break

                yolo_labels = []
                if os.path.isfile(xml_path):
                    voc_objs = voc_to_yolo(xml_path)
                    for cname, x, y, w, h in voc_objs:
                        if cname not in class_to_idx:
                            class_to_idx[cname] = len(class_to_idx)
                            classes.append(cname)
                            print(f"Added new class '{cname}' as index {class_to_idx[cname]}")
                        cls_idx = class_to_idx[cname]
                        yolo_labels.append(f"{cls_idx} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                dst_lbl = os.path.join(out_labels, os.path.splitext(new_img_name)[0] + ".txt")
                with open(dst_lbl, "w", encoding="utf-8") as f:
                    if yolo_labels:
                        f.write("\n".join(yolo_labels))
                    else:
                        f.write("")
                all_image_paths.append(dst_img)
    else:
        print("VOC root path not found:", voc_root)

    # 4) make splits
    random.shuffle(all_image_paths)
    n = len(all_image_paths)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_paths = all_image_paths[:n_train]
    val_paths = all_image_paths[n_train:n_train + n_val]
    test_paths = all_image_paths[n_train + n_val:]

    def write_list(filelist, out_txt):
        with open(out_txt, "w", encoding="utf-8") as f:
            for p in filelist:
                f.write(os.path.abspath(p) + "\n")

    write_list(train_paths, os.path.join(output_dir, "train.txt"))
    write_list(val_paths, os.path.join(output_dir, "val.txt"))
    write_list(test_paths, os.path.join(output_dir, "test.txt"))

    # 5) create data.yaml
    data_yaml_path = os.path.join(output_dir, "data.yaml")
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        f.write(f"train: {os.path.abspath(os.path.join(output_dir, 'train.txt'))}\n")
        f.write(f"val: {os.path.abspath(os.path.join(output_dir, 'val.txt'))}\n")
        f.write(f"test: {os.path.abspath(os.path.join(output_dir, 'test.txt'))}\n\n")
        f.write(f"nc: {len(classes)}\n")
        f.write("names: [")
        f.write(", ".join([f'\"{n}\"' for n in classes]))
        f.write("]\n")

    print("Merged dataset created at:", output_dir)
    print(f"Images: {len(all_image_paths)}, classes: {classes}")
    print("data.yaml written to", data_yaml_path)
    print("train/val/test split sizes:", len(train_paths), len(val_paths), len(test_paths))


# -------------------------
# Hardcoded paths (your requested locations)
# -------------------------
if __name__ == "__main__":
    # set these to your exact folders (forward slashes or raw strings are fine)
    YOLO_A_PATH = r"E:/e_storage/dataset/New pothole detection.v2i.yolov12"
    VOC_ROOT   = r"E:/e_storage/dataset/archive (6)/train"
    PBTXT_PATH = r"E:/e_storage/dataset/archive (6)/train/label_map.pbtxt"
    OUTPUT_DIR = r"E:/e_storage/dataset/merged_pothole"

    # optional: override classes (if you know there is only one class)
    # CLASSES = ["pothole"]
    CLASSES = None

    # merge (will create OUTPUT_DIR)
    merge_datasets(
        yolo_A_path=YOLO_A_PATH,
        voc_root=VOC_ROOT,
        output_dir=OUTPUT_DIR,
        pbtxt_path=PBTXT_PATH,
        classes_override=CLASSES,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        seed=42
    )
