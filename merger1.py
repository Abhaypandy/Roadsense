#!/usr/bin/env python3
"""
merger.py (updated)

Supports explicit separate YOLO images/labels folders via --yolo_images and --yolo_labels.

Usage example (PowerShell):
python "E:\e_storage\Roadsense\merger.py" --yolo_images "E:/e_storage/dataset/New pothole detection.v2i.yolov12/train/images" --yolo_labels "E:/e_storage/dataset/New pothole detection.v2i.yolov12/test/labels" --voc_root "E:/e_storage/dataset/archive (6)/train" --pbtxt "E:/e_storage/dataset/archive (6)/train/label_map.pbtxt" --output "E:/e_storage/dataset/merged_pothole"
"""

import os
import shutil
import xml.etree.ElementTree as ET
import random
import argparse
from typing import List

# ---------- helpers ----------
def parse_pbtxt(pbtxt_path: str) -> List[str]:
    names = []
    if not os.path.isfile(pbtxt_path):
        return names
    with open(pbtxt_path, "r", encoding="utf-8") as f:
        text = f.read()
    for block in text.split("item"):
        if "name" in block:
            try:
                part = block.split("name",1)[1]
                if "'" in part:
                    name = part.split("'")[1]
                elif '"' in part:
                    name = part.split('"')[1]
                else:
                    name = part.strip().split()[0]
                if name:
                    names.append(name)
            except Exception:
                continue
    return names

def voc_to_yolo(xml_path: str):
    try:
        tree = ET.parse(xml_path)
    except Exception:
        return []
    root = tree.getroot()
    size = root.find("size")
    if size is None:
        return []
    try:
        width = float(size.find("width").text)
        height = float(size.find("height").text)
    except Exception:
        return []
    objs = []
    for obj in root.findall("object"):
        name_node = obj.find("name")
        bnd = obj.find("bndbox")
        if name_node is None or bnd is None:
            continue
        try:
            name = name_node.text.strip()
            xmin = float(bnd.find("xmin").text); ymin = float(bnd.find("ymin").text)
            xmax = float(bnd.find("xmax").text); ymax = float(bnd.find("ymax").text)
        except Exception:
            continue
        x_center = ((xmin + xmax) / 2.0) / width
        y_center = ((ymin + ymax) / 2.0) / height
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        w = max(0.0, min(1.0, w)); h = max(0.0, min(1.0, h))
        objs.append((name, x_center, y_center, w, h))
    return objs

# ---------- merge ----------
def merge_datasets(yolo_images_dir, yolo_labels_dir, voc_root, output_dir, pbtxt_path=None, classes_override=None,
                   train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    out_images = os.path.join(output_dir, "images"); os.makedirs(out_images, exist_ok=True)
    out_labels = os.path.join(output_dir, "labels"); os.makedirs(out_labels, exist_ok=True)

    # class list
    classes = []
    if classes_override:
        classes = classes_override
    elif pbtxt_path and os.path.isfile(pbtxt_path):
        parsed = parse_pbtxt(pbtxt_path)
        if parsed:
            classes = parsed
    if not classes:
        classes = ["pothole"]

    class_to_idx = {c: i for i, c in enumerate(classes)}
    print("[*] Using classes:", classes)

    all_image_paths = []

    # === handle YOLO images/labels if provided ===
    if yolo_images_dir and os.path.isdir(yolo_images_dir):
        print(f"[*] Found YOLO images dir: {yolo_images_dir}")
        # labels dir can be provided separately; if not, try sibling labels dir
        if not yolo_labels_dir or not os.path.isdir(yolo_labels_dir):
            candidate = os.path.join(os.path.dirname(yolo_images_dir), "labels")
            if os.path.isdir(candidate):
                yolo_labels_dir = candidate
        if yolo_labels_dir and os.path.isdir(yolo_labels_dir):
            print(f"[*] Using YOLO labels dir: {yolo_labels_dir}")
        else:
            print("[!] YOLO labels dir not provided or not found. Will create empty labels for YOLO images if missing.")

        for fn in sorted(os.listdir(yolo_images_dir)):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            src_img = os.path.join(yolo_images_dir, fn)
            prefix = "A_"
            new_name = prefix + fn
            dst_img = os.path.join(out_images, new_name)
            shutil.copy2(src_img, dst_img)
            # corresponding label
            base = os.path.splitext(fn)[0]; src_lbl = None
            if yolo_labels_dir:
                alt = os.path.join(yolo_labels_dir, base + ".txt")
                if os.path.isfile(alt):
                    src_lbl = alt
            dst_lbl = os.path.join(out_labels, os.path.splitext(new_name)[0] + ".txt")
            if src_lbl:
                shutil.copy2(src_lbl, dst_lbl)
            else:
                open(dst_lbl, "w").close()
            all_image_paths.append(dst_img)
    else:
        print("[!] No YOLO images dir provided or not found.")

    # === convert VOC subfolders ===
    if not os.path.isdir(voc_root):
        print(f"[!] VOC root not found: {voc_root}")
    else:
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
                        images_dir = os.path.join(subdir, d); break
            if not xml_dir:
                for root, _, files in os.walk(subdir):
                    if any(f.lower().endswith(".xml") for f in files):
                        xml_dir = root; break
            if not os.path.isdir(images_dir) or not os.path.isdir(xml_dir):
                print(f"[!] Skipping {sub}: images or xmls not found. images_dir={images_dir}, xml_dir={xml_dir}")
                continue
            print(f"[*] Processing VOC subfolder: {sub} (images: {images_dir}, xmls: {xml_dir})")
            for img_fn in sorted(os.listdir(images_dir)):
                if not img_fn.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                src_img = os.path.join(images_dir, img_fn)
                new_img_name = f"{sub}_{img_fn}"
                dst_img = os.path.join(out_images, new_img_name)
                shutil.copy2(src_img, dst_img)
                base = os.path.splitext(img_fn)[0]; xml_path = os.path.join(xml_dir, base + ".xml")
                if not os.path.isfile(xml_path):
                    for f in os.listdir(xml_dir):
                        if os.path.splitext(f)[0].lower() == base.lower() and f.lower().endswith(".xml"):
                            xml_path = os.path.join(xml_dir, f); break
                yolo_lines = []
                if os.path.isfile(xml_path):
                    voc_objs = voc_to_yolo(xml_path)
                    for cname, x, y, w, h in voc_objs:
                        if cname not in class_to_idx:
                            class_to_idx[cname] = len(class_to_idx); classes.append(cname)
                            print(f"  [*] Added new class '{cname}' as index {class_to_idx[cname]}")
                        idx = class_to_idx[cname]
                        yolo_lines.append(f"{idx} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                dst_lbl = os.path.join(out_labels, os.path.splitext(new_img_name)[0] + ".txt")
                with open(dst_lbl, "w", encoding="utf-8") as f:
                    if yolo_lines: f.write("\n".join(yolo_lines))
                all_image_paths.append(dst_img)

    # ---- make splits ----
    if len(all_image_paths) == 0:
        print("[!] No images collected. Exiting.")
        return
    random.shuffle(all_image_paths)
    n = len(all_image_paths); n_train = int(n * train_ratio); n_val = int(n * val_ratio)
    train_paths = all_image_paths[:n_train]; val_paths = all_image_paths[n_train:n_train+n_val]; test_paths = all_image_paths[n_train+n_val:]
    def write_list(lst, fname):
        with open(fname, "w", encoding="utf-8") as f:
            for p in lst: f.write(os.path.abspath(p) + "\n")
    write_list(train_paths, os.path.join(output_dir, "train.txt"))
    write_list(val_paths, os.path.join(output_dir, "val.txt"))
    write_list(test_paths, os.path.join(output_dir, "test.txt"))

    # write data.yaml
    data_yaml_path = os.path.join(output_dir, "data.yaml")
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        f.write(f"train: {os.path.abspath(os.path.join(output_dir,'train.txt'))}\n")
        f.write(f"val:   {os.path.abspath(os.path.join(output_dir,'val.txt'))}\n")
        f.write(f"test:  {os.path.abspath(os.path.join(output_dir,'test.txt'))}\n\n")
        f.write(f"nc: {len(classes)}\n")
        f.write("names: [")
        f.write(", ".join([f'\"{n}\"' for n in classes]))
        f.write("]\n")
    print("\n[+] Merged dataset created at:", output_dir)
    print("    Images:", len(all_image_paths))
    print("    Classes:", classes)
    print("    train/val/test sizes:", len(train_paths), len(val_paths), len(test_paths))
    print("    data.yaml ->", data_yaml_path)

# ---------- CLI ----------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--yolo_images", help="Path to YOLO images folder (e.g. .../train/images)")
    p.add_argument("--yolo_labels", help="Path to YOLO labels folder (e.g. .../train/labels or .../test/labels)")
    p.add_argument("--voc_root", required=True, help="Root of VOC-style archive (contains Czech/India/...)")
    p.add_argument("--output", required=True, help="Output merged folder")
    p.add_argument("--pbtxt", help="Path to label_map.pbtxt")
    p.add_argument("--classes", nargs="+", help="Optional override classes")
    p.add_argument("--train", type=float, default=0.7)
    p.add_argument("--val", type=float, default=0.2)
    p.add_argument("--test", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    s = args.train + args.val + args.test
    if abs(s - 1.0) > 1e-6:
        args.train /= s; args.val /= s; args.test /= s
        print("[!] Normalized train/val/test ratios to sum to 1")

    merge_datasets(yolo_images_dir=args.yolo_images, yolo_labels_dir=args.yolo_labels,
                   voc_root=args.voc_root, output_dir=args.output,
                   pbtxt_path=args.pbtxt, classes_override=args.classes,
                   train_ratio=args.train, val_ratio=args.val, test_ratio=args.test,
                   seed=args.seed)

if __name__ == "__main__":
    main()
