from ultralytics import YOLO

model= YOLO(f'yolo11x-seg.pt', task='segment')
results= model.train(data=f'data.yaml',epochs=300, imgsz=640, dropout= 0.4, device=[0,1,2,3], patience= 15,
                    lr0=0.0005,## Réduit le taux d'apprentissage final
                    lrf=0.01,  # Ajustement du facteur de réduction de la LR
                    momentum=0.937,#influençant l'incorporation des gradients passés dans la mise à jour actuelle.
                    weight_decay=0.01,
                    box=0.5,#Facteur de pondération pour la perte de localisation.
                    cls=0.5,# Facteur de pondération pour la perte de classification.
                    optimizer="SGD",
                    hsv_h=0.015,  # Variations de teinte
                    hsv_s=0.7,    # Saturation
                    hsv_v=0.4,    # Valeur
                    degrees=10.0,
                    bgr=0.2,copy_paste=0.35, mosaic= 0.75, mixup= 0.3, dfl= 10, label_smoothing= .5, translate=.4, 
                    flipud= .35,
                    cache=True )

