import json
import os
import random
import shutil
import yaml
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

def split_coco_annotations(coco_json_path, images_dir, output_dir, train_ratio=0.8, seed=42):
    """
    Divise les annotations COCO en ensembles d'entraînement et de validation.
    """
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    with open(coco_json_path, 'r') as f:
        coco = json.load(f)

    images = coco.get('images', [])
    annotations = coco.get('annotations', [])
    categories = coco.get('categories', [])

    if not images or not annotations or not categories:
        raise ValueError("Le fichier COCO annotations.json doit contenir 'images', 'annotations' et 'categories'.")

    image_ids = [img['id'] for img in images]
    random.shuffle(image_ids)

    num_train = int(len(image_ids) * train_ratio)
    train_ids = set(image_ids[:num_train])
    val_ids = set(image_ids[num_train:])

    # Séparer les images
    train_images = [img for img in images if img['id'] in train_ids]
    val_images = [img for img in images if img['id'] in val_ids]

    # Séparer les annotations
    train_annotations = [ann for ann in annotations if ann['image_id'] in train_ids]
    val_annotations = [ann for ann in annotations if ann['image_id'] in val_ids]

    # Copier les images
    for subset, subset_images in zip(['train', 'val'], [train_images, val_images]):
        for img in subset_images:
            src = os.path.join(images_dir, img['file_name'])
            dst = os.path.join(output_dir, subset, 'images', img['file_name'])
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(src, dst)

    # Créer les nouveaux fichiers COCO annotations
    for subset, subset_annotations, subset_images in zip(
        ['train', 'val'],
        [train_annotations, val_annotations],
        [train_images, val_images]
    ):
        subset_coco = {
            'images': subset_images,
            'annotations': subset_annotations,
            'categories': categories
        }

        # Ajouter 'info' et 'licenses' si elles existent
        if 'info' in coco:
            subset_coco['info'] = coco['info']

        if 'licenses' in coco:
            subset_coco['licenses'] = coco['licenses']

        with open(os.path.join(output_dir, subset, 'annotations.json'), 'w') as f:
            json.dump(subset_coco, f)

        print(f"{subset.capitalize()} set: {len(subset_images)} images")

# Mapping explicite des classes
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

def create_category_mapping(coco_json_path):
    """
    Crée un mapping de category_id COCO vers class_id basé sur le class_mapping fourni.
    """
    with open(coco_json_path, 'r') as f:
        coco = json.load(f)

    categories = coco.get('categories', [])
    if not categories:
        raise ValueError("Le fichier COCO annotations.json doit contenir 'categories'.")

    # Utiliser le class_mapping explicite
    category_id_map = {}
    category_names = []
    for category in categories:
        cat_id = category['id']
        cat_name = category['name']
        
        # Rechercher dans le class_mapping
        for class_id, name in class_mapping.items():
            if name == cat_name:
                category_id_map[cat_id] = class_id - 1  # -1 pour l'indexation YOLO (commence à 0)
                category_names.append(name)
                break

    if not category_id_map:
        raise ValueError("Aucun mapping trouvé entre les catégories COCO et le class_mapping.")

    return category_id_map, category_names


def split_annotations_segmentation(coco_json_path, output_labels_dir, category_id_map):
    """
    Convertit les annotations COCO en fichiers d'annotations YOLOv8 pour la segmentation.
    """
    with open(coco_json_path, 'r') as f:
        coco = json.load(f)

    images = coco.get('images', [])
    annotations = coco.get('annotations', [])

    # Créer un mapping image_id -> annotations
    image_id_to_annotations = {}
    for ann in annotations:
        image_id = ann['image_id']
        if image_id not in image_id_to_annotations:
            image_id_to_annotations[image_id] = []
        image_id_to_annotations[image_id].append(ann)

    # Écrire les annotations individuelles dans des fichiers .txt
    for img in images:
        img_id = img['id']
        base_name = os.path.splitext(img['file_name'])[0]
        label_path = os.path.join(output_labels_dir, f"{base_name}.txt")
        os.makedirs(os.path.dirname(label_path), exist_ok=True)

        img_width = img['width']
        img_height = img['height']

        anns = image_id_to_annotations.get(img_id, [])

        with open(label_path, 'w') as f_out:
            if not anns:
                print(f"Aucune annotation trouvée pour l'image {img['file_name']}. Création d'un label vide.")
            else:
                for ann in anns:
                    # Obtenir le category_id et vérifier qu'il est mappé
                    category_id = ann.get('category_id')
                    if category_id is None:
                        raise ValueError(f"L'annotation pour l'image_id {img_id} ne contient pas 'category_id'.")
                    if category_id not in category_id_map:
                        raise ValueError(f"Unknown `category_id` {category_id} dans l'annotation.")

                    class_id = category_id_map[category_id]

                    bbox = ann.get('bbox')
                    if bbox is None or len(bbox) != 4:
                        raise ValueError(f"BBox invalide pour l'annotation dans image_id {img_id}.")

                    # Format COCO : [x_min, y_min, width, height] en pixels
                    x_min, y_min, bbox_width, bbox_height = bbox

                    # Ajustement des bounding boxes pour rester dans les dimensions de l'image
                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(x_min + bbox_width, img_width)
                    y_max = min(y_min + bbox_height, img_height)
                    bbox_width = x_max - x_min
                    bbox_height = y_max - y_min

                    # Calculer les coordonnées du centre et normaliser
                    x_center = (x_min + bbox_width / 2) / img_width
                    y_center = (y_min + bbox_height / 2) / img_height
                    width_norm = bbox_width / img_width
                    height_norm = bbox_height / img_height

                    # Vérifier que les valeurs sont entre 0 et 1
                    if not all(0 <= val <= 1 for val in [x_center, y_center, width_norm, height_norm]):
                        print(f"** Valeurs hors de la plage [0, 1] pour l'image {img['file_name']}: x_center={x_center}, y_center={y_center}, width={width_norm}, height={height_norm}")
                        raise ValueError(f"Valeurs de bbox normalisées hors limites pour l'image {img['file_name']}.")

                    # Traitement de la segmentation
                    segmentation = ann.get('segmentation')
                    if segmentation is None:
                        raise ValueError(f"L'annotation pour l'image_id {img_id} ne contient pas 'segmentation'.")

                    polygon = []
                    if isinstance(segmentation, list):
                        # Format polygone
                        for seg in segmentation:
                            # Chaque 'seg' est une liste de points [x1, y1, x2, y2, ..., xn, yn]
                            xs = seg[0::2]
                            ys = seg[1::2]
                            # Normaliser les points
                            normalized_points = []
                            for x, y in zip(xs, ys):
                                x_norm = x / img_width
                                y_norm = y / img_height
                                normalized_points.extend([x_norm, y_norm])
                            polygon.extend(normalized_points)
                    else:
                        raise ValueError(f"Format de segmentation non supporté pour image_id {img_id}.")

                    # Vérifier que le polygone n'est pas vide
                    if not polygon:
                        raise ValueError(f"Le polygone est vide pour l'image_id {img_id}.")

                    # Écrire les valeurs normalisées dans le fichier .txt
                    label_line = [str(class_id), f"{x_center:.6f}", f"{y_center:.6f}", f"{width_norm:.6f}", f"{height_norm:.6f}"]
                    label_line.extend([f"{pt:.6f}" for pt in polygon])
                    f_out.write(' '.join(label_line) + '\n')

                    print(f"Écriture dans {label_path}: {' '.join(label_line)}")

    print(f"Annotations de segmentation converties en fichiers .txt dans '{output_labels_dir}'.")

def create_data_yaml(train_dir, val_dir, category_names, output_path='data.yaml'):
    data = {
        'train': os.path.abspath(os.path.join(train_dir, '..')),  # Pointer vers le dossier parent des images
        'val': os.path.abspath(os.path.join(val_dir, '..')),
        'nc': len(category_names),
        'names': category_names
    }

    with open(output_path, 'w') as f:
        yaml.dump(data, f)

    print(f"Fichier data.yaml créé à {output_path}")

def main_pipeline():
    # Chemins des données
    original_coco_json_path = 'dataset/annotation/annotation-corrigé-sans-tree-shadow-velux-coco.json'
    images_dir = 'dataset/images'
    split_output_dir = 'dataset_split'

    # Étape 1 : Diviser les données en train et val
    split_coco_annotations(
        coco_json_path=original_coco_json_path,
        images_dir=images_dir,
        output_dir=split_output_dir,
        train_ratio=0.8,
        seed=42
    )

    # Étape 2 : Créer le mapping des category_id à partir des annotations d'entraînement
    train_annotations_path = os.path.join(split_output_dir, 'train', 'annotations.json')
    category_id_map, category_names = create_category_mapping(train_annotations_path)

    # Étape 3 : Convertir les annotations en fichiers individuels pour train et val avec segmentation
    for subset in ['train', 'val']:
        annotations_path = os.path.join(split_output_dir, subset, 'annotations.json')
        labels_dir = os.path.join(split_output_dir, subset, 'labels')
        split_annotations_segmentation(
            coco_json_path=annotations_path,
            output_labels_dir=labels_dir,
            category_id_map=category_id_map
        )

    # Étape 4 : Créer data.yaml avec les annotations ajustées
    create_data_yaml(
        train_dir=os.path.join(split_output_dir, 'train', 'images'),
        val_dir=os.path.join(split_output_dir, 'val', 'images'),
        category_names=category_names,
        output_path='data.yaml'
    )

    # Étape 5 : Instancier et entraîner le modèle YOLOv8 pour segmentation
    model = YOLO('yolov8n-seg.pt')
    model.train(data='data.yaml', epochs=2, imgsz=1024,device=[0,1,2,3])

if __name__ == "__main__":
    main_pipeline()
