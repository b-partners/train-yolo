import os
from ultralytics import YOLO

# Charger un modèle YOLOv8 avec segmentation
model = YOLO("runs/segment/train2/weights/best.pt")
"""
# Configurer et effectuer la validation
validation_results = model.val(data="data.yaml",   # Chemin vers le fichier de configuration des données
    imgsz=1024,         # Taille des images d'entrée
    batch=16,           # Taille de batch
    conf=0.25,          # Seuil de confiance
    iou=0.7
)"""

# Validate the model
metrics = model.val(data="data.yaml",conf=0.90)  

metrics.box.maps



"""# Afficher les résultats
print("Résultats de la validation :")
print(validation_results)"""
