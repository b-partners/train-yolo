import os
import json
from ultralytics import YOLO
import cv2
import glob
from post_processing_yolo import *
from convert_vgg_to_coco import *
from convert_yolo_to_vgg import *

# Set PyTorch CUDA allocation configuration
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Load YOLOv8 model with segmentation
model = YOLO("best-weight-prediction/best.pt")

# Output directory for annotations
output_dir = "runs/output_test_images"
os.makedirs(output_dir, exist_ok=True)

# Prepare the list of valid image files
image_files = glob.glob('test/dijon/images/*.*')

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
output_vgg_file = os.path.join(output_dir, "dijon_vgg_annotations.json")
yolo_results_to_vgg(all_results, class_mapping, output_vgg_file)

#post-traitement des prédictions 
# Charger les annotations VGG générées
#data = load_vgg_annotations(output_vgg_file)


# Fusionner les masques superposés de même classe
#data = merge_overlapping_masks(data)

# Effectuer le post-traitement des masques (lissage, suppression des petits masques, etc.)
#data = post_process_masks(data)


# Enregistrer les annotations post-traitées dans un nouveau fichier JSON
#post_processed_file = os.path.join(output_dir, "dijon_vgg_annotations_post_trait.json")
#save_vgg_annotations(data, post_processed_file)

#print(f"Post-traitement terminé. Fichier enregistré dans : {post_processed_file}")

#convert vgg to coco 

coco_json_path = os.path.join(output_dir,'dijon_coco_annotations.json')
convert_vgg_to_coco(output_vgg_file, coco_json_path)

