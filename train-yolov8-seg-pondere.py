from ultralytics import YOLO

# Charger le modèle
model = YOLO("yolov8x-seg.pt")

# Multiplier chaque poids par 5
for param in model.model.parameters():
    param.data *= 5

# Entraîner le modèle
train_results = model.train(
    data="data.yaml",  # chemin vers le fichier YAML du dataset
    epochs=100,        # nombre d'époques d'entraînement
    imgsz=1024,        # taille des images d'entraînement
    device=[0,1,2,3]   # appareils à utiliser pour l'entraînement
)

# Charger le meilleur modèle entraîné
model = YOLO("runs/segment/train3/weights/best.pt")

# Effectuer la détection d'objets sur une image
results = model("test_images", save=True)
