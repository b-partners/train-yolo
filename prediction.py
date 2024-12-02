import os
import json
from ultralytics import YOLO
import cv2
import glob

# Set PyTorch CUDA allocation configuration
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

def yolo_results_to_vgg(results, class_mapping, output_file):
    """
    Convert YOLOv8 segmentation results to VGG JSON format with confidence scores.

    Args:
        results (list): YOLOv8 results containing polygons, class information, and scores.
        class_mapping (dict): Mapping of class indices to class labels.
        output_file (str): Path to save the VGG JSON file.
    """
    vgg_data = {}

    for result in results:
        if not hasattr(result, "path") or not os.path.exists(result.path):
            print(f"Warning: Missing or invalid path for result {result}. Skipping.")
            continue

        image_filename = os.path.basename(result.path)
        vgg_entry = {
            "fileref": "",
            "size": os.path.getsize(result.path),
            "filename": image_filename,
            "base64_img_data": "",
            "file_attributes": {},
            "regions": {}
        }

        # Check if masks and boxes are available
        if result.masks is None or result.boxes is None:
            print(f"No masks or boxes detected for image: {image_filename}.")
            vgg_data[image_filename] = vgg_entry
            continue

        # Iterate over each detection
        region_index = 0
        for segment, cls, conf in zip(result.masks.xy, result.boxes.cls, result.boxes.conf):
            cls_index = int(cls)  # YOLO class indices start from 0
            label = class_mapping.get(cls_index, f"class_{cls_index}")
            confidence = float(conf)  # Convert confidence score to float

            all_points_x = segment[:, 0].tolist()
            all_points_y = segment[:, 1].tolist()

            vgg_entry["regions"][str(region_index)] = {
                "shape_attributes": {
                    "name": "polygon",
                    "all_points_x": all_points_x,
                    "all_points_y": all_points_y
                },
                "region_attributes": {
                    "label": label,
                    "confidence": confidence
                }
            }
            region_index += 1

        vgg_data[image_filename] = vgg_entry

    # Save to output file
    with open(output_file, "w") as json_file:
        json.dump(vgg_data, json_file, indent=4)
    print(f"VGG annotations with confidence scores saved to {output_file}")


# Load YOLOv8 model with segmentation
model = YOLO("best-weight-prediction/best.pt")

# Output directory for annotations
output_dir = "runs/output_test_images"
os.makedirs(output_dir, exist_ok=True)

# Prepare the list of valid image files
image_files = glob.glob('test/beziers/*.*')

# Filter out non-image files or corrupted images
valid_image_files = []
for img_path in image_files:
    try:
        img = cv2.imread(img_path)
        if img is not None:
            valid_image_files.append(img_path)
        else:
            print(f"Invalid or corrupted image file: {img_path}. Skipping.")
    except Exception as e:
        print(f"Error reading image {img_path}: {e}. Skipping.")

if not valid_image_files:
    print("No valid images found in the directory.")
    exit(1)

# Class mapping (adjust according to your model)
class_mapping = {
    0: 'line',
    1: 'pathway',
    2: 'roof_tuiles',
    3: 'roof_beton',
    4: 'sidewalk',
    5: 'roof_ardoise',
    6: 'solar_panel',
    7: 'pool',
    8: 'roof_autres',
    9: 'parking',
    10: 'green_space'
}

# Initialize an empty list to store all results
all_results = []

# Process images one at a time
for img_path in valid_image_files:
    try:
        result = model.predict(
            source=img_path,
            show_conf=True,
            conf=0.25,
            save=True,
            device=0
        )
        all_results.extend(result)
    except Exception as e:
        print(f"Error during model prediction for {img_path}: {e}")
        continue

# Generate VGG JSON file
output_vgg_file = os.path.join(output_dir, "beziers_vgg_annotations.json")
yolo_results_to_vgg(all_results, class_mapping, output_vgg_file)
