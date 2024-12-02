
import cv2
import json
import os
import numpy as np

#rotated_-90_0a2d8c0c-34dd-4a80-92df-a19502dc07ce
# Chargement des annotations depuis un fichier JSON
with open('/Users/bparners/Desktop/projet-reference-git-azur/image-processing-reference-pathway/samples/features/dataset/vgg_annotations-pathway-mrcnn.json', 'r') as json_file:
    annotations = json.load(json_file)

# Dossier contenant les images
dossier_images = '/Users/bparners/Desktop/projet-reference-git-azur/image-processing-reference-pathway/samples/features/dataset/test'  # Mettez le chemin de votre dossier d'images
dossier_images_annotes = '/Users/bparners/Desktop/projet-reference-git-azur/image-processing-reference-pathway/samples/features/dataset/test-mrcnn-pathway-check'  # Créez un nouveau dossier pour enregistrer les images annotées

# Créez le dossier pour les images annotées s'il n'existe pas déjà
if not os.path.exists(dossier_images_annotes):
    os.makedirs(dossier_images_annotes)

# Limite le nombre d'images à traiter (dans cet exemple, nous allons traiter les 100 premières images)
limite_images = 100000
images_traitees = 0

# Créez une fenêtre personnalisée pour afficher les images annotées avec une taille personnalisée
cv2.namedWindow('Image avec Annotations', cv2.WINDOW_NORMAL)

# Définir la taille personnalisée de la fenêtre (largeur, hauteur)
cv2.resizeWindow('Image avec Annotations', 800, 600)  # Ajustez les valeurs selon vos préférences

# Parcourir les fichiers d'images dans le dossier
for image_filename in os.listdir(dossier_images):
    if image_filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):  # Filtrer les fichiers d'images
        # Charger l'image
        image_path = os.path.join(dossier_images, image_filename)
        image = cv2.imread(image_path)
        

        if image is not None:
            
            # Rechercher les annotations correspondantes dans les données JSON
            #print(image_path)
            if image_filename in annotations:
                print(image_filename)
                image_annotations = annotations[image_filename]
                for region_id, region_info in image_annotations['regions'].items():
                    shape_attributes = region_info['shape_attributes']
                    label = region_info['region_attributes']['label']

                    # Extraire les points du polygone
                    all_points_x = shape_attributes['all_points_x']
                    all_points_y = shape_attributes['all_points_y']

                    # Convertir les points du polygone en liste de tuples
                    polygon_points = [(x, y) for x, y in zip(all_points_x, all_points_y)]
                    # Convertissez la liste de points en un tableau numpy avec le type CV_32S
                    polygon_points_array = np.array(polygon_points, dtype=np.int32)

                    # Dessiner le polygone sur l'image
                    #cv2.polylines(image, [np.array(polygon_points)], isClosed=True, color=(0, 255, 0), thickness=2)
                    cv2.polylines(image, [polygon_points_array], isClosed=True, color=(0, 255, 0), thickness=2)


                    
                    
                    # Définir la taille de la police et la couleur du texte
                    font_scale = 0.5  # Ajustez cette valeur pour la taille de la police
                    font_color = (255, 255, 255)  # Couleur du texte en RVB (blanc ici)

                    # Définir la position du texte au centre du polygone
                    text_x = int(np.mean(all_points_x))
                    text_y = int(np.mean(all_points_y))
                    text_position = (text_x, text_y)

                    # Utiliser la police HERSHEY_SIMPLEX
                    font = cv2.FONT_HERSHEY_COMPLEX_SMALL

                    # Afficher le nom de la catégorie au centre du polygone
                    cv2.putText(image, label, text_position, font, font_scale, font_color, thickness=1, lineType=cv2.LINE_AA)    

                # Afficher l'image avec annotation dans la fenêtre personnalisée
                cv2.imshow('Image avec Annotations', image)

                # Enregistrer l'image annotée dans le nouveau dossier avec les dimensions de la fenêtre personnalisée
                image_annotee_path = os.path.join(dossier_images_annotes, image_filename)
                cv2.imwrite(image_annotee_path, image)


            # Incrémentez le compteur d'images traitées
            images_traitees += 1

            # Vérifiez si le nombre limite d'images a été atteint
            if images_traitees >= limite_images:
                break
print(images_traitees)
# Fermez la fenêtre OpenCV après avoir traité les 100 premières images
cv2.destroyAllWindows()