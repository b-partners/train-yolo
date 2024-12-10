import cv2
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
    if image is None:
        print(f"Erreur : l'image {image_path} n'a pas pu être chargée.")
        return
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Lire les annotations
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])
            polygon_points = list(map(float, parts[1:]))

            # Convertir les coordonnées normalisées en pixels
            img_height, img_width, _ = image.shape
            polygon_points = [p * img_width if i % 2 == 0 else p * img_height for i, p in enumerate(polygon_points)]

            # Vérifier que le nombre de points est pair
            if len(polygon_points) % 2 != 0:
                print(f"Erreur : le nombre de coordonnées dans l'annotation est impair dans {label_path}.")
                continue

            # Dessiner le polygone
            pts = np.array(polygon_points).reshape(-1, 2).astype(np.int32)
            cv2.polylines(image, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

            # Calculer le centre du polygone pour placer le label
            center_x = int(np.mean(pts[:, 0]))
            center_y = int(np.mean(pts[:, 1]))

            # Ajouter le nom du label au centre du polygone
            label_name = class_mapping.get(class_id, 'Unknown')
            cv2.putText(image, label_name, (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

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
output_dir='augmented_dataset/'

process_directory(
    images_dir='images/',
    labels_dir='labels/',
    output_dir='train-images-check'
)
