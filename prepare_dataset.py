import os
import random
import shutil
import xml.etree.ElementTree as ET

# =========================
# CONFIGURATION
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGE_DIR = os.path.join(BASE_DIR, "IMAGES")
ANNOTATION_DIR = os.path.join(BASE_DIR, "ANNOTATIONS")
OUTPUT_DIR = os.path.join(BASE_DIR, "YOLO_DATASET")

CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]

# Reproducible split
random.seed(42)

TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10


# =========================
# CREATE DIRECTORIES
# =========================

for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_DIR, "images", split), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "labels", split), exist_ok=True)


# =========================
# FIND DATASET FILES
# =========================

image_files = [
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(".jpg")
]

print(f"Found {len(image_files)} images.")


# =========================
# SHUFFLE DATASET
# =========================

random.shuffle(image_files)

total = len(image_files)

train_end = int(total * TRAIN_RATIO)
val_end = train_end + int(total * VAL_RATIO)

train_files = image_files[:train_end]
val_files = image_files[train_end:val_end]
test_files = image_files[val_end:]

splits = {
    "train": train_files,
    "val": val_files,
    "test": test_files
}


# =========================
# XML → YOLO CONVERSION
# =========================

def convert_annotation(xml_path):

    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")

    image_width = int(size.find("width").text)
    image_height = int(size.find("height").text)

    yolo_annotations = []

    for obj in root.findall("object"):

        class_name = obj.find("name").text.strip()

        if class_name not in CLASSES:
            print(f"WARNING: Unknown class {class_name}")
            continue

        class_id = CLASSES.index(class_name)

        bbox = obj.find("bndbox")

        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        # Convert Pascal VOC → YOLO
        x_center = ((xmin + xmax) / 2) / image_width
        y_center = ((ymin + ymax) / 2) / image_height

        width = (xmax - xmin) / image_width
        height = (ymax - ymin) / image_height

        yolo_annotations.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} "
            f"{width:.6f} {height:.6f}"
        )

    return yolo_annotations


# =========================
# PROCESS DATASET
# =========================

for split_name, files in splits.items():

    print(f"\nProcessing {split_name}: {len(files)} images")

    for image_file in files:

        image_path = os.path.join(IMAGE_DIR, image_file)

        annotation_file = os.path.splitext(image_file)[0] + ".xml"

        annotation_path = os.path.join(
            ANNOTATION_DIR,
            annotation_file
        )

        if not os.path.exists(annotation_path):
            print(f"WARNING: Missing annotation for {image_file}")
            continue

        # Destination image
        destination_image = os.path.join(
            OUTPUT_DIR,
            "images",
            split_name,
            image_file
        )

        shutil.copy2(image_path, destination_image)

        # Convert annotation
        yolo_annotations = convert_annotation(annotation_path)

        label_file = os.path.splitext(image_file)[0] + ".txt"

        label_path = os.path.join(
            OUTPUT_DIR,
            "labels",
            split_name,
            label_file
        )

        with open(label_path, "w") as f:
            f.write("\n".join(yolo_annotations))


# =========================
# CREATE data.yaml
# =========================

yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")

with open(yaml_path, "w") as f:

    f.write(f"path: {OUTPUT_DIR}\n")
    f.write("train: images/train\n")
    f.write("val: images/val\n")
    f.write("test: images/test\n\n")

    f.write("names:\n")

    for i, class_name in enumerate(CLASSES):
        f.write(f"  {i}: {class_name}\n")


# =========================
# FINAL SUMMARY
# =========================

print("\n===================================")
print("DATASET PREPARATION COMPLETE")
print("===================================")

print(f"Total images : {total}")
print(f"Training     : {len(train_files)}")
print(f"Validation   : {len(val_files)}")
print(f"Testing      : {len(test_files)}")

print(f"\nDataset created at:")
print(OUTPUT_DIR)

print("\nClasses:")

for i, class_name in enumerate(CLASSES):
    print(f"{i}: {class_name}")