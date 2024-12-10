import json
import networkx as nx
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from collections import defaultdict
# Paramètres globaux
TOLERANCE = 5.0  # Tolérance pour le lissage des contours
AREA_THRESHOLD = 1000  # Seuil de surface pour supprimer les petits masques
EXCLUDED_CLASSES = ["roof_tuiles", "roof_ardoise", "roof_beton", "roof_autres"]

def load_vgg_annotations(filepath):
    """Charge les annotations VGG depuis un fichier JSON."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def save_vgg_annotations(data, filepath):
    """Enregistre les annotations VGG dans un fichier JSON."""
    with open(filepath, 'w') as f:
        json.dump(data, f)

def merge_overlapping_masks(data):
    """Fusionne les masques superposés de même classe dans les annotations VGG,
    sauf pour les classes spécifiées dans EXCLUDED_CLASSES."""
    for image_key, image_data in data.items():
        regions = image_data.get('regions', {})
        # Regrouper les régions par label (classe)
        label_to_regions = defaultdict(list)
        for region_id, region_data in regions.items():
            label = region_data['region_attributes'].get('label')
            if label:
                label_to_regions[label].append((region_id, region_data))

        new_regions = {}
        region_count = 0
        # Pour chaque label, traiter les régions correspondantes
        for label, regions_list in label_to_regions.items():
            # Vérifier si la classe est exclue
            if label in EXCLUDED_CLASSES:
                # Ajouter directement les régions sans modification
                for region_id, region_data in regions_list:
                    new_regions[str(region_count)] = region_data
                    region_count += 1
                continue  # Passer à l'itération suivante pour éviter la fusion

            # Créer des polygones à partir des régions
            polygons = []
            confidences = []
            region_attrs_list = []
            for idx, (region_id, region_data) in enumerate(regions_list):
                shape_attr = region_data['shape_attributes']
                if shape_attr['name'] == 'polygon':
                    x_points = shape_attr['all_points_x']
                    y_points = shape_attr['all_points_y']
                    points = list(zip(x_points, y_points))
                    polygon = Polygon(points)
                    if not polygon.is_valid:
                        polygon = polygon.buffer(0)  # Corriger les polygones invalides
                    polygons.append(polygon)
                    # Récupérer la confiance, en la convertissant en float
                    confidence = float(region_data['region_attributes'].get('confidence', 0))
                    confidences.append(confidence)
                    region_attrs_list.append(region_data['region_attributes'])

            # Construire un graphe des chevauchements
            G = nx.Graph()
            for idx1, poly1 in enumerate(polygons):
                G.add_node(idx1)
                for idx2 in range(idx1 + 1, len(polygons)):
                    poly2 = polygons[idx2]
                    if poly1.intersects(poly2):
                        G.add_edge(idx1, idx2)

            # Trouver les composantes connexes (groupes de polygones qui se chevauchent)
            components = list(nx.connected_components(G))

            # Traiter chaque composante
            for component in components:
                component_polygons = [polygons[idx] for idx in component]
                # Fusionner les polygones dans la composante
                merged_polygon = unary_union(component_polygons)
                # Trouver le masque avec la confiance la plus élevée
                max_confidence = -1
                max_idx = None
                for idx in component:
                    conf = confidences[idx]
                    if conf > max_confidence:
                        max_confidence = conf
                        max_idx = idx
                # Obtenir les attributs du masque avec la plus haute confiance
                max_region_attributes = region_attrs_list[max_idx]

                # Gérer les MultiPolygons
                if isinstance(merged_polygon, MultiPolygon):
                    polys = merged_polygon.geoms
                else:
                    polys = [merged_polygon]

                for poly in polys:
                    if poly.is_empty:
                        continue
                    x, y = poly.exterior.coords.xy
                    all_points_x = list(x)
                    all_points_y = list(y)
                    shape_attributes = {
                        'name': 'polygon',
                        'all_points_x': all_points_x,
                        'all_points_y': all_points_y
                    }
                    region_attributes = {
                        'label': label
                        # Copier d'autres attributs si nécessaire
                    }
                    new_region = {
                        'shape_attributes': shape_attributes,
                        'region_attributes': region_attributes
                    }
                    new_regions[str(region_count)] = new_region
                    region_count += 1

        # Mettre à jour les régions de l'image avec les nouvelles régions fusionnées
        image_data['regions'] = new_regions

    return data



def post_process_masks(data, tolerance=TOLERANCE, area_threshold=AREA_THRESHOLD):
    """Effectue le post-traitement des masques : lissage, remplissage des trous, suppression des petits masques."""
    for image_key, image_data in data.items():
        regions = image_data.get('regions', {})
        new_regions = {}
        region_count = 0

        for region_id, region_data in regions.items():
            label = region_data['region_attributes'].get('label')
            # Ne pas traiter les classes exclues
            """if label in EXCLUDED_CLASSES:
                new_regions[str(region_count)] = region_data
                region_count += 1
                continue    
            """
            # Lissage et suppression des petits masques
            shape_attr = region_data['shape_attributes']
            if shape_attr['name'] == 'polygon':
                x_points = shape_attr['all_points_x']
                y_points = shape_attr['all_points_y']
                points = list(zip(x_points, y_points))
                polygon = Polygon(points)
                if not polygon.is_valid:
                    polygon = polygon.buffer(0)
                if polygon.area < area_threshold:
                    continue
                polygon = polygon.simplify(tolerance, preserve_topology=True)
                if not polygon.is_valid or polygon.is_empty:
                    continue

                # Gérer les MultiPolygons
                if isinstance(polygon, MultiPolygon):
                    # Fusionner les polygones dans un seul (union) ou sélectionner le plus grand
                    polygons = list(polygon.geoms)  # Extraire les polygones individuels
                    polygon = max(polygons, key=lambda p: p.area)  # Sélectionner le plus grand polygone

                if not polygon.is_valid or polygon.is_empty:
                    continue

                # Extraire les coordonnées du polygone
                x, y = polygon.exterior.coords.xy
                shape_attributes = {
                    'name': 'polygon',
                    'all_points_x': list(x),
                    'all_points_y': list(y)
                }
                region_data['shape_attributes'] = shape_attributes
                new_regions[str(region_count)] = region_data
                region_count += 1

        # Mettre à jour les régions de l'image
        image_data['regions'] = new_regions

    return data


def filter_by_confidence(data, class_thresholds, default_threshold=0.0):
    """
    Ajoute les annotations qui satisfont les seuils de confiance spécifiques à chaque classe.

    Args:
        data (dict): Annotations VGG au format JSON.
        class_thresholds (dict): Dictionnaire des seuils de confiance par classe (index numérique).
        default_threshold (float): Seuil de confiance par défaut pour les classes non définies.

    Returns:
        dict: Annotations mises à jour avec les annotations supplémentaires filtrées.
    """
    for image_key, image_data in data.items():
        regions = image_data.get('regions', {})
        new_regions = regions.copy()  # Conserver les annotations existantes
        region_count = len(new_regions)  # Commencer à partir du nombre existant d'annotations

        for region_id, region_data in regions.items():
            label = region_data['region_attributes'].get('label')
            confidence = float(region_data['region_attributes'].get('confidence', 0))

            # Obtenir l'index de la classe depuis le label (supposé être convertible en entier)
            try:
                class_index = int(label)  # Si le label est un index numérique
            except ValueError:
                # Si le label n'est pas un index valide, ignorer l'annotation
                #print(f"Annotation ignorée pour label non valide : {label}")
                continue

            # Obtenir le seuil pour la classe ou utiliser le seuil par défaut
            threshold = class_thresholds.get(class_index, default_threshold)

            # Ajouter uniquement les annotations qui dépassent le seuil de confiance
            if confidence >= threshold:
                new_regions[str(region_count)] = region_data
                region_count += 1
           

        # Mettre à jour les régions avec les nouvelles annotations filtrées
        image_data['regions'] = new_regions

    return data
