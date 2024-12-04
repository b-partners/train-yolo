import os
import json
from shapely.geometry import Polygon
import uuid

def convert_vgg_to_coco(vgg_json_path, coco_json_path, default_width=1024, default_height=1024):
    """
    Convertit un fichier d'annotations VGG en format COCO.
    """
    def calculate_bbox(polygon_points):
        """Calculate the bounding box of a polygon."""
        if len(polygon_points['all_points_x']) < 3:
            return None  # Invalid polygon
        if (polygon_points['all_points_x'][0] != polygon_points['all_points_x'][-1] or
            polygon_points['all_points_y'][0] != polygon_points['all_points_y'][-1]):
            polygon_points['all_points_x'].append(polygon_points['all_points_x'][0])
            polygon_points['all_points_y'].append(polygon_points['all_points_y'][0])
        polygon = Polygon(zip(polygon_points['all_points_x'], polygon_points['all_points_y']))
        min_x, min_y, max_x, max_y = polygon.bounds
        return min_x, min_y, max_x - min_x, max_y - min_y

    def calculate_area(polygon_points):
        """Calculate the area of a polygon."""
        if len(polygon_points['all_points_x']) < 3:
            return 0  # Invalid polygon
        if (polygon_points['all_points_x'][0] != polygon_points['all_points_x'][-1] or
            polygon_points['all_points_y'][0] != polygon_points['all_points_y'][-1]):
            polygon_points['all_points_x'].append(polygon_points['all_points_x'][0])
            polygon_points['all_points_y'].append(polygon_points['all_points_y'][0])
        polygon = Polygon(zip(polygon_points['all_points_x'], polygon_points['all_points_y']))
        return polygon.area

    # Vérifier et créer le répertoire parent du fichier COCO
    coco_dir = os.path.dirname(coco_json_path)
    if not os.path.exists(coco_dir):
        os.makedirs(coco_dir)

    # Lecture du fichier VGG
    with open(vgg_json_path) as file:
        vgg_data = json.load(file)

    # Initialisation de la structure de base COCO
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Dictionnaire pour les identifiants uniques des catégories
    category_ids = {}

    # Traitement de chaque image dans les données VGG
    for filename, image_info in vgg_data.items():
        image_id = str(uuid.uuid4())  # ID d'image unique
        coco_data['images'].append({
            "id": image_id,
            "width": image_info.get("width", default_width),
            "height": image_info.get("height", default_height),
            "file_name": filename
        })

        for region in image_info.get('regions', {}).values():
            category_name = region['region_attributes']['label']
            if category_name not in category_ids:
                category_id = str(uuid.uuid4())  # ID unique pour la catégorie
                category_ids[category_name] = category_id
                coco_data['categories'].append({
                    "id": category_id,
                    "name": category_name
                })
            else:
                category_id = category_ids[category_name]

            polygon_points = region['shape_attributes']
            bbox = calculate_bbox(polygon_points)
            if not bbox:  # Ignorer les polygones invalides
                continue
            area = calculate_area(polygon_points)

            segmentation = []
            all_x = polygon_points['all_points_x']
            all_y = polygon_points['all_points_y']
            for point in range(len(all_x)):
                segmentation.append(all_x[point])
                segmentation.append(all_y[point])

            coco_data['annotations'].append({
                "id": str(uuid.uuid4()),
                "segmentation": [segmentation],
                "area": area,
                "iscrowd": 0,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": bbox
            })

    # Sauvegarde des données COCO dans un fichier
    with open(coco_json_path, 'w') as file:
        json.dump(coco_data, file, indent=4)

    print(f"Conversion terminée. Les données COCO sont enregistrées dans : {coco_json_path}")
