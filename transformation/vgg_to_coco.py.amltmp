import json
import os
from shapely.geometry import Polygon
import uuid  # Pour générer des identifiants uniques

# Chemin du fichier VGG
# Define the relative path to the JSON file from the 'transformation' directory
vgg_json_path = '../runs/output_test_images/beziers_vgg_annotations.json'

# Optionally, print out the absolute path for verification
vgg_json_file_absolute_path = os.path.abspath(vgg_json_path)
print("JSON file absolute path:", vgg_json_file_absolute_path)

#cocofile 
coco_json_path = '../runs/output_test_images/beziers_coco_annotations.json'

# Optionally, print out the absolute path for verification
coco_json_file_absolute_path = os.path.abspath(coco_json_path)
print("JSON file absolute path:", coco_json_file_absolute_path)




# Lecture du fichier VGG
with open(vgg_json_file_absolute_path) as file:
    vgg_data = json.load(file)

# Initialisation de la structure de base COCO
coco_data = {
    "images": [],
    "annotations": [],
    "categories": []
}

# Un dictionnaire pour garder la trace des identifiants de catégorie uniques
category_ids = {}
def calculate_bbox(polygon_points):
    """
    Calculate the bounding box of a polygon given by its points.
    """
    if len(polygon_points['all_points_x']) < 3:
        # Return a default or error value; here we return None to indicate an error
        return None
    # Ensure the polygon closes by checking if the first and last points are the same
    if (polygon_points['all_points_x'][0] != polygon_points['all_points_x'][-1] or
        polygon_points['all_points_y'][0] != polygon_points['all_points_y'][-1]):
        polygon_points['all_points_x'].append(polygon_points['all_points_x'][0])
        polygon_points['all_points_y'].append(polygon_points['all_points_y'][0])
    
    polygon = Polygon(zip(polygon_points['all_points_x'], polygon_points['all_points_y']))
    min_x, min_y, max_x, max_y = polygon.bounds
    return min_x, min_y, max_x - min_x, max_y - min_y

def calculate_area(polygon_points):
    """
    Calculate the area of a polygon.
    """
    if len(polygon_points['all_points_x']) < 3:
        return 0  # Return zero area for invalid polygons
    if (polygon_points['all_points_x'][0] != polygon_points['all_points_x'][-1] or
        polygon_points['all_points_y'][0] != polygon_points['all_points_y'][-1]):
        polygon_points['all_points_x'].append(polygon_points['all_points_x'][0])
        polygon_points['all_points_y'].append(polygon_points['all_points_y'][0])
    
    polygon = Polygon(zip(polygon_points['all_points_x'], polygon_points['all_points_y']))
    return polygon.area

# Traitement de chaque image dans les données VGG
for filename, image_info in vgg_data.items():
    image_id = str(uuid.uuid4())  # Générer un ID d'image unique
    coco_data['images'].append({
        "id": image_id,
        "width": image_info.get("width", 1024),
        "height": image_info.get("height", 1024),
        "file_name": filename
    })
    for region in image_info['regions'].values():
        segmentation = []  # Réinitialisation pour chaque région
        category_name = region['region_attributes']['label']
        if category_name not in category_ids:
            category_id = str(uuid.uuid4())  # Générer un ID de catégorie unique
            category_ids[category_name] = category_id
            coco_data['categories'].append({
                "id": category_id,
                "name": category_name
            })
        else:
            category_id = category_ids[category_name]

        polygon_points = region['shape_attributes']
        bbox = calculate_bbox(polygon_points)
        area = calculate_area(polygon_points)
        all_x = polygon_points['all_points_x']
        all_y = polygon_points['all_points_y']
        for point in range(len(all_x)):
            segmentation.append(all_x[point])
            segmentation.append(all_y[point])
        
        coco_data['annotations'].append({
            "id": str(uuid.uuid4()),
            "segmentation": [segmentation],
            "area": area,
            "iscrowd": False,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": bbox
        })

# Sauvegarde des données COCO dans un nouveau fichier
with open(coco_json_file_absolute_path, 'w') as file:
    json.dump(coco_data, file, indent=4)

print("Conversion terminée.")
