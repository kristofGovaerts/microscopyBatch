#DOES NOT WORK YET

import json
import os
from PIL import Image
import numpy as np
import shutil
from detectron2.structures import BoxMode
from sklearn.model_selection import train_test_split

# Set the paths to your COCO annotations and image directories
coco_annotations_file = r"C:\Users\govaerts.kristof\OneDrive - GroupeFD\Desktop\temp\temp\Fodder Beets/via_project_5May2023_14h12m_coco.json"
image_directory = r"C:\Users\govaerts.kristof\OneDrive - GroupeFD\Desktop\temp\temp\Fodder Beets\images"
detectron2_directory = r"C:\Users\govaerts.kristof\OneDrive - GroupeFD\Desktop\temp\temp\Fodder Beets"


# Set the train/test/val fractions
train_fraction = 0.7
test_fraction = 0.15
val_fraction = 0.15

# Load the COCO annotations
with open(coco_annotations_file, "r") as f:
    coco_annotations = json.load(f)

# Split the images into train/test/val
image_filenames = [os.path.join(image_directory, image["file_name"]) for image in coco_annotations["images"]]
train_filenames, test_val_filenames = train_test_split(image_filenames, test_size=(test_fraction+val_fraction))
test_filenames, val_filenames = train_test_split(test_val_filenames, test_size=(val_fraction/(test_fraction+val_fraction)))

# Create the 'coco', 'img', and 'ann' directories
os.makedirs(os.path.join(detectron2_directory, "coco"))
os.makedirs(os.path.join(detectron2_directory, "img"))
os.makedirs(os.path.join(detectron2_directory, "ann"))

# Iterate over each image in the COCO annotations
for image in coco_annotations["images"]:
    # Determine which folder this image should go in (train, test, or val)
    if os.path.join(image_directory, image["file_name"]) in train_filenames:
        folder = "train"
    elif os.path.join(image_directory, image["file_name"]) in test_filenames:
        folder = "test"
    else:
        folder = "val"

    # Copy the image file to the 'img' directory
    shutil.copyfile(os.path.join(image_directory, image["file_name"]), os.path.join(detectron2_directory, "img", image["file_name"]))

    # Create a new Detectron2 annotation dictionary for this image
    annotation_dict = {
        "file_name": image["file_name"],
        "image_id": image["id"],
        "height": image["height"],
        "width": image["width"],
        "annotations": []
    }

    # Find all the annotations for this image and add them to the annotation_dict
    annotations = [a for a in coco_annotations["annotations"] if a["image_id"] == image["id"]]
    for annotation in annotations:
        bbox = annotation["bbox"]
        category_id = annotation["category_id"]
        annotation_dict["annotations"].append({
            "bbox": [bbox[0], bbox[1], bbox[2], bbox[3]],
            "bbox_mode": 0,
            "category_id": category_id
        })

    # Save the annotation_dict to a JSON file in the appropriate folder in the 'ann' directory
    annotation_filename = os.path.join(detectron2_directory, "ann", folder, image["file_name"].replace(".jpg", ".json"))
    with open(annotation_filename, "w") as f:
        json.dump(annotation_dict, f)

train_coco = {
    "info": coco_annotations["info"],
    "licenses": coco_annotations["licenses"],
    "images": [image for image in coco_annotations["images"] if os.path.join(image_directory, image["file_name"]) in train_filenames],
    "annotations": [annotation for annotation in coco_annotations["annotations"] if annotation["image_id"] in [image["id"] for image in train_coco["images"]]],
    "categories": coco_annotations["categories"]
}
train_coco_filename = os.path.join(detectron2_directory, "coco", "train.json")
with open(train_coco_filename, "w") as f:
    json.dump(train_coco, f)

test_coco = {
    "info": coco_annotations["info"],
    "licenses": coco_annotations["licenses"],
    "images": [image for image in coco_annotations["images"] if os.path.join(image_directory, image["file_name"]) in test_filenames],
    "annotations": [annotation for annotation in coco_annotations["annotations"] if annotation["image_id"] in [image["id"] for image in test_coco["images"]]],
    "categories": coco_annotations["categories"]
}
test_coco_filename = os.path.join(detectron2_directory, "coco", "test.json")
with open(test_coco_filename, "w") as f:
    json.dump(test_coco, f)

val_coco = {
    "info": coco_annotations["info"],
    "licenses": coco_annotations["licenses"],
    "images": [image for image in coco_annotations["images"] if os.path.join(image_directory, image["file_name"]) in val_filenames],
    "annotations": [annotation for annotation in coco_annotations["annotations"] if annotation["image_id"] in [image["id"] for image in val_coco["images"]]],
    "categories": coco_annotations["categories"]
}
val_coco_filename = os.path.join(detectron2_directory, "coco", "val.json")
with open(val_coco_filename, "w") as f:
    json.dump(val_coco, f)

