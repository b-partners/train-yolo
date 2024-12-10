# Output directory for annotations
output_dir = "runs/output_test_images"
os.makedirs(output_dir, exist_ok=True)

output_vgg_file = os.path.join(output_dir, "beziers_vgg_annotations.json")
#post-traitement des prédictions 
# Charger les annotations VGG générées
data = load_vgg_annotations(output_vgg_file)

# Effectuer le post-traitement des masques (lissage, suppression des petits masques, etc.)
data = post_process_masks(data)

# Fusionner les masques superposés de même classe
data = merge_overlapping_masks(data)

# Filtrer les annotations par seuil de confiance
data = filter_by_confidence(data, class_thresholds)



# Enregistrer les annotations post-traitées dans un nouveau fichier JSON
post_processed_file = os.path.join(output_dir, "beziers_vgg_annotations_post_traits_seuil_optimal.json")
save_vgg_annotations(data, post_processed_file)

print(f"Post-traitement terminé. Fichier enregistré dans : {post_processed_file}")

#convert vgg to coco 

coco_json_path = os.path.join(output_dir,'beziers_coco_annotations_post_traits_seuil_optimal.json')
convert_vgg_to_coco(post_processed_file, coco_json_path)