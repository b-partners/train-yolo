import cv2
import json
import os
import numpy as np

# Define the relative path to the JSON file from the 'transformation' directory
json_file_path = '../runs/output_test_images/beziers_vgg_annotations.json'

# Optionally, print out the absolute path for verification
json_file_absolute_path = os.path.abspath(json_file_path)
print("JSON file absolute path:", json_file_absolute_path)

# Check if the file exists before attempting to open it
if not os.path.exists(json_file_path):
    print(f"File not found: {json_file_path}")
    exit(1)

# Open the JSON file
with open(json_file_path, 'r') as json_file:
    annotations = json.load(json_file)

# Define paths for the image directories
dossier_images = '../test/beziers'
dossier_images_path = os.path.abspath(dossier_images)
print("dossier_images absolute path:", dossier_images_path)

dossier_images_annotes = '../test/beziers-check'
dossier_images_annotes_path = os.path.abspath(dossier_images_annotes)
print("dossier_images_annotes absolute path:", dossier_images_annotes_path)

# Create the directory for annotated images if it doesn't exist
if not os.path.exists(dossier_images_annotes):
    os.makedirs(dossier_images_annotes)

# Limit the number of images to process
limite_images = 100000
images_traitees = 0

# Process each image file in the directory
for image_filename in os.listdir(dossier_images):
    if image_filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):  # Filter image files
        # Load the image
        image_path = os.path.join(dossier_images, image_filename)
        image = cv2.imread(image_path)

        if image is not None:
            # Search for corresponding annotations in the JSON data
            if image_filename in annotations:
                print(f"Processing: {image_filename}")
                image_annotations = annotations[image_filename]
                regions = image_annotations.get('regions', {})
                for region_id, region_info in regions.items():
                    shape_attributes = region_info['shape_attributes']
                    label = region_info['region_attributes']['label']

                    # Extract polygon points
                    all_points_x = shape_attributes['all_points_x']
                    all_points_y = shape_attributes['all_points_y']

                    # Convert polygon points to a NumPy array
                    polygon_points = np.array(list(zip(all_points_x, all_points_y)), dtype=np.int32)

                    # Draw the polygon on the image
                    cv2.polylines(image, [polygon_points], isClosed=True, color=(0, 255, 0), thickness=2)

                    # Define font settings
                    font_scale = 0.5
                    font_color = (255, 255, 255)  # White color in BGR
                    font = cv2.FONT_HERSHEY_COMPLEX_SMALL

                    # Calculate the position for the label text
                    text_x = int(np.mean(all_points_x))
                    text_y = int(np.mean(all_points_y))
                    text_position = (text_x, text_y)

                    # Display the label at the center of the polygon
                    cv2.putText(image, label, text_position, font, font_scale, font_color, thickness=1, lineType=cv2.LINE_AA)

                # Save the annotated image to the new directory
                image_annotee_path = os.path.join(dossier_images_annotes, image_filename)
                cv2.imwrite(image_annotee_path, image)

            else:
                print(f"No annotations found for image: {image_filename}")

            # Increment the processed images counter
            images_traitees += 1

            # Check if the image limit has been reached
            if images_traitees >= limite_images:
                break

print(f"Total images processed: {images_traitees}")
