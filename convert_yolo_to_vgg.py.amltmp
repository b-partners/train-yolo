import os
import json
import cv2
import glob
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

