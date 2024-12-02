# ---- train-yolo
il faut uploader : yolo11n-seg.pt ou yolo8n-seg.pt 
installer : ultralytics et pytorch 

# ---- le train et la détection sont  établis sur 4 gpu

pour commencer : 
1/ uploader votre dataset : train et val , dont la structure doit étre : 
train 
   ---images
   ---labels 
val
   ---images
   ---labels
labels doit contenir un fichier contenant les annotations en format txt et non pas vgg 

# -----  pour convertir le vegg , il faut utiliser le dossier : transformation/vgg_to_txt_yolo.py

2/ lancer le train : 
activer la machine : compute-v100-gpu
dans le console : aller dans le dossier avec "cd" 
                  puis tapez : conda activate myenv  
                               python train.py

le output tu le trouve dans le dossier runs/segment/train

3/pour la prédiction : 
créer un dossier avec les images dont laquelles tu veux faire tes prédictions
puis, dans le console : tapez prediction.py

le output tu le trouve dans le dossier runs/segment/predict
(si nécéssaire changer dans le code pour l'adapter à ton besoin)

4/pour l'évaluation: 
dans le console : tapez evaluation.py

le output tu le trouve dans le dossier runs/segment/val

