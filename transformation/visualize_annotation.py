import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# Mapping des classes
class_mapping = {
    1: 'line',
    2: 'pathway',
    3: 'roof_tuiles',
    4: 'roof_beton',
    5: 'sidewalk',
    6: 'roof_ardoise',
    7: 'solar_panel',
    8: 'pool',
    9: 'roof_autres',
    10: 'parking',
    11: 'green_space'
}

def visualize_and_save_annotations(image_path, label_path, output_path):
    # Lire l'image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Lire les annotations
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:5])
            polygon_points = list(map(float, parts[5:]))

            # Convertir les coordonnées normalisées en pixels
            img_height, img_width, _ = image.shape
            x_center *= img_width
            y_center *= img_height
            width *= img_width
            height *= img_height
            polygon_points = [p * img_width if i % 2 == 0 else p * img_height for i, p in enumerate(polygon_points)]

            # Dessiner le polygone
            pts = np.array(polygon_points).reshape(-1, 2).astype(np.int32)
            cv2.polylines(image, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

            # Ajouter le nom du label sur l'image
            label_name = class_mapping.get(class_id, 'Unknown')
            text_position = (int(x_center - width / 2), int(y_center - height / 2))
            cv2.putText(image, label_name, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Déterminer le chemin de sortie et sauvegarder l'image
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_file_path = os.path.join(output_path, f"{base_name}_annotated.jpg")
    os.makedirs(output_path, exist_ok=True)

    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_file_path, image_bgr)
    print(f"Image sauvegardée à : {output_file_path}")

# Parcourir un dossier d'images et sauvegarder les outputs dans un autre dossier
def process_directory(images_dir, labels_dir, output_dir):
    for image_filename in os.listdir(images_dir):
        if image_filename.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(images_dir, image_filename)
            label_path = os.path.join(labels_dir, os.path.splitext(image_filename)[0] + '.txt')
            visualize_and_save_annotations(image_path, label_path, output_dir)

# Exemple d'utilisation
process_directory(
    images_dir='dataset_split/val/images',
    labels_dir='dataset_split/val/labels',
    output_dir='output_images/annotated2'
)
