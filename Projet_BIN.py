# ============================================================
# PROJET BIN — Segmentation de fonds de rivière
# Pipeline : COCO JSON → Masques → Augmentation → Export
# ============================================================

import os
import json
import copy
import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image
from pycocotools import mask as coco_mask
import albumentations as A

# ============================================================
# 1. CHARGEMENT DU JSON ET STRUCTURES DE BASE
# ============================================================

# RLE = au lieu de stocker chaque pixel individuellement,
# on stocke des séquences alternées "N pixels à 0, N pixels à 1"
# [3996686, 7, 2992, 12, ...]
#  ↑ 3.9M pixels à 0  ↑ 7 pixels à 1  ↑ 2992 à 0  ↑ 12 à 1...

with open('annotations_clean.json', 'r') as f:
    data = json.load(f)

images_dict = {img['id']: img for img in data['images']}
categories  = {cat['id']: cat['name'] for cat in data['categories']}

# print("Catégories :", categories)
# print(f"Images      : {len(data['images'])}")
# print(f"Annotations : {len(data['annotations'])}")

# ============================================================
# 2. DÉCODAGE RLE ET CONSTRUCTION DU MASQUE SÉMANTIQUE
# ============================================================

# CVAT exporte le RLE avec 'counts' comme liste d'entiers
# pycocotools attend 'counts' comme bytes encodés
# → on utilise frPyObjects pour convertir avant decode()

def construire_masque_semantique(image_id, data, images_dict, categories):
    img_info      = images_dict[image_id]
    h, w          = img_info['height'], img_info['width']
    masque_global = np.zeros((h, w), dtype=np.uint8)

    anns_image = [a for a in data['annotations']
                  if a['image_id'] == image_id]

    for ann in anns_image:
        rle    = coco_mask.frPyObjects(ann['segmentation'], h, w)
        masque = coco_mask.decode(rle)

        if masque.ndim == 3:
            masque = masque[:, :, 0]

        masque_global[masque == 1] = ann['category_id']  # ← dans la boucle 

    return masque_global, img_info

# ============================================================
# 3. NETTOYAGE DES CATÉGORIES
# ============================================================

# Suppression du label "pebble" original et fusion gravel → pebble
# 1. Renommer gravel → pebble
# 2. Supprimer l'ancienne catégorie pebble (id=2, 0 annotations)
# 3. Supprimer les annotations de l'ancien pebble

def nettoyer_categories(json_source, json_sortie):
    with open(json_source, 'r') as f:
        d = json.load(f)

    id_gravel = next(c['id'] for c in d['categories'] if c['name'] == 'gravel')
    id_pebble = next(c['id'] for c in d['categories'] if c['name'] == 'pebble')

    # Supprimer ancien pebble, renommer gravel → pebble
    d['categories'] = [c for c in d['categories']
                       if c['name'] != 'pebble' or c['id'] == id_gravel]
    for cat in d['categories']:
        if cat['id'] == id_gravel:
            cat['name'] = 'pebble'

    # Supprimer annotations de l'ancien pebble
    d['annotations'] = [a for a in d['annotations']
                        if a['category_id'] != id_pebble]

    with open(json_sortie, 'w') as f:
        json.dump(d, f)

    print(f"Catégories nettoyées → {json_sortie}")
    for cat in d['categories']:
        print(f"  id={cat['id']} — {cat['name']}")

# ============================================================
# 4. PIPELINE D'AUGMENTATION
# ============================================================

# Albumentations applique la MÊME transformation aléatoire
# simultanément à l'image ET au masque
# → indispensable pour garder la cohérence spatiale

# p=0.5 = probabilité que la transformation soit appliquée
# → chaque augmentation est différente car tirée aléatoirement

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussNoise(p=0.2),
])

"""
Le RandomCrop (p=0.5) élimine beaucoup d'annotations quand les objets sortent du cadre. C'est particulièrement visible sur vegetation :
vegetation : 1079 → 209  ← perte de 81% des annotations
pebble     :  561 → 328  ← perte de 42%
target     :   45 → 81   ← augmentation (petits objets, restent dans le crop)
Deux options pour corriger ça :
Option A : Réduire p du RandomCrop (0.5 → 0.2)
           → moins de crops agressifs, plus d'annotations conservées

Option B : Augmenter la taille minimale du crop (2048 → 2800)
           → crop moins agressif sur des images 3000×4000

On a fait l'option B — un crop de 2800×2800 sur une image 3000×4000 conserve 87% de la surface :

Resultat : 
Le problème fondamental est que les grandes zones de végétation (qui couvrent parfois 80% de l'image) sont encodées en un seul masque RLE — quand le crop les touche même légèrement, le masque résultant est souvent vide ou très petit et passe sous le seuil sum() == 0.
La vraie solution est de supprimer complètement le RandomCrop — les autres augmentations (flip, rotation, luminosité) sont suffisantes et ne perdent aucune annotation. On supprime RandomCrop du pipeline.
"""

# ============================================================
# 5. EXPORT DU DATASET AUGMENTÉ AU FORMAT COCO JSON
# ============================================================

def masque_vers_rle(masque_binaire):
    """Convertit un masque numpy BINAIRE en RLE — version vectorisée"""
    # Aplatir en ordre Fortran (convention COCO)
    flat = masque_binaire.flatten(order='F').astype(np.uint8)

    # Trouver les positions où la valeur change
    changements = np.where(np.diff(flat))[0] + 1

    # Construire les counts
    positions = np.concatenate([[0], changements, [len(flat)]])
    counts    = np.diff(positions).tolist()

    # Si le masque commence par un 1, ajouter un 0 au début
    if flat[0] == 1:
        counts.insert(0, 0)

    return {
        'counts': counts,
        'size'  : [masque_binaire.shape[0], masque_binaire.shape[1]]
    }

def exporter_dataset_augmente(data, images_dict, categories,
                               transform,
                               n_augmentations=5,
                               dossier_images_src='images',
                               dossier_sortie='dataset_augmente'):
    """
    Génère le dataset augmenté complet et exporte :
    - Les images augmentées dans dossier_sortie/images/
    - Le fichier COCO JSON dans dossier_sortie/annotations.json

    Annotations perdues si RandomCrop sort l'objet du cadre
    → masque_binaire.sum() == 0 → annotation ignorée
    """
    os.makedirs(f'{dossier_sortie}/images', exist_ok=True)

    nouveau_json = {
        'licenses'   : data.get('licenses', []),
        'info'       : data.get('info', {}),
        'categories' : data['categories'],
        'images'     : [],
        'annotations': []
    }

    image_id_counter = 1
    ann_id_counter   = 1

    for img_original in data['images']:
        print(f"{img_original['file_name']}", end=' ')

        img_path = os.path.join(dossier_images_src, img_original['file_name'])
        if not os.path.exists(img_path):
            print("non trouvée")
            continue

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        masque_sem, _ = construire_masque_semantique(
            img_original['id'], data, images_dict, categories
        )

        # Augmentation adaptative : plus d'augmentations
        # pour les images contenant des classes rares (pebble)
        images_avec_pebble = set(
            a['image_id'] for a in data['annotations']
            if a['category_id'] == 1
        )
        n_aug = 15 if img_original['id'] in images_avec_pebble \
                else n_augmentations
        print(f"→ {n_aug} augmentations")

        for aug_idx in range(n_aug):
            augmented  = transform(image=img, mask=masque_sem)
            img_aug    = augmented['image']
            masque_aug = augmented['mask']
            h, w       = img_aug.shape[:2]

            nom_base = os.path.splitext(img_original['file_name'])[0]
            nom_aug  = f"{nom_base}_aug{aug_idx:03d}.jpg"

            cv2.imwrite(
                os.path.join(dossier_sortie, 'images', nom_aug),
                cv2.cvtColor(img_aug, cv2.COLOR_RGB2BGR)
            )

            nouveau_json['images'].append({
                'id'       : image_id_counter,
                'file_name': nom_aug,
                'width'    : w,
                'height'   : h
            })

            for cat in data['categories']:
                cat_id         = cat['id']
                masque_binaire = (masque_aug == cat_id).astype(np.uint8)

                if masque_binaire.sum() == 0:
                    continue

                rle  = masque_vers_rle(masque_binaire)
                area = float(masque_binaire.sum())

                rows = np.any(masque_binaire, axis=1)
                cols = np.any(masque_binaire, axis=0)
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                bbox = [float(cmin), float(rmin),
                        float(cmax-cmin), float(rmax-rmin)]

                nouveau_json['annotations'].append({
                    'id'          : ann_id_counter,
                    'image_id'    : image_id_counter,
                    'category_id' : cat_id,
                    'segmentation': rle,
                    'area'        : area,
                    'bbox'        : bbox,
                    'iscrowd'     : 1
                })
                ann_id_counter += 1

            image_id_counter += 1

    with open(f'{dossier_sortie}/annotations.json', 'w') as f:
        json.dump(nouveau_json, f)

    print(f"\nDataset exporté dans '{dossier_sortie}/'")
    print(f"   Images      : {len(nouveau_json['images'])}")
    print(f"   Annotations : {len(nouveau_json['annotations'])}")

    from collections import Counter
    distrib = Counter(a['category_id'] for a in nouveau_json['annotations'])
    print(f"\nDistribution finale :")
    for cat_id, count in sorted(distrib.items()):
        print(f"  {categories[cat_id]:<15} : {count:>4} annotations")





# ============================================================
# Import de 2 datasets d'entraînement (train_set1.zip et train_set2.zip)
# ============================================================

import zipfile
import os

# def deziper_et_inspecter(zip_ path, dossier_sortie):
#     """Dézipe une archive et affiche son contenu"""
#     print(f"\n Dézippage de {zip_path}...")

#     with zipfile.ZipFile(zip_path, 'r') as z:
#         z.extractall(dossier_sortie)

#     print(f"Extrait dans '{dossier_sortie}/'")
#     print(f"\nContenu :")

#     for root, dirs, files in os.walk(dossier_sortie):
#         niveau = root.replace(dossier_sortie, '').count(os.sep)
#         indent = '  ' * niveau
#         print(f"{indent} {os.path.basename(root)}/")
#         for f in sorted(files):
#             taille = os.path.getsize(os.path.join(root, f)) / (1024**2)
#             print(f"{indent}  - {f} ({taille:.1f} MB)")

# # Dézipper les deux datasets
# deziper_et_inspecter('train_set1.zip', 'train_set1')
# deziper_et_inspecter('train_set2.zip', 'train_set2')



# ============================================================
# Comparaison des catégories et distribution des annotations dans les 3 JSON :
# ============================================================


# # Inspecter les catégories des 3 JSON
# fichiers = [
#     ('Dataset original', 'annotations_clean.json'),
#     ('Train set 1',      'train_set1/train_set1/annotations/instances_default.json'),
#     ('Train set 2',      'train_set2/train_set2/annotations/instances_default.json'),
# ]

# for nom, chemin in fichiers:
#     with open(chemin, 'r') as f:
#         d = json.load(f)
#     print(f"\n {nom}")
#     print(f"   Images      : {len(d['images'])}")
#     print(f"   Annotations : {len(d['annotations'])}")
#     print(f"   Catégories  :")
#     for cat in d['categories']:
#         n_ann = sum(1 for a in d['annotations'] if a['category_id'] == cat['id'])
#         print(f"     id={cat['id']} — {cat['name']:<15} ({n_ann} annotations)")


"""On a 3 catégories différentes dans les 3 JSON :
- Dataset original : pebble (id=2), vegetation (id=3), gravel (id=4)
- Train set 1      : pebble (id=1), vegetation (id=2), target (id=3)
- Train set 2      : pebble (id=1), vegetation (id=2), target (id=3)
Le dataset original a une catégorie "gravel" qui correspond à "pebble" dans les autres sets.
Il faut donc :
1. Fusionner gravel → pebble dans le dataset original
2. Réassigner les IDs pour que pebble=1, vegetation=2, target=3 dans tous les datasets
3. Supprimer les annotations de catégories inconnues (ex : target dans le dataset original)
"""


# ============================================================
# Comparaison des catégories et distribution des annotations dans les 3 JSON :
# ============================================================


CATEGORIES_CIBLES = [
    {'id': 1, 'name': 'pebble'},
    {'id': 2, 'name': 'vegetation'},
    {'id': 3, 'name': 'target'},
]

def normaliser_dataset(chemin_json, dossier_images):
    """
    Normalise un dataset COCO vers le schéma de catégories cible.
    - Fusionne gravel → pebble
    - Réassigne les IDs selon CATEGORIES_CIBLES
    - Supprime les annotations de catégories inconnues
    """
    with open(chemin_json, 'r') as f:
        d = json.load(f)

    # Construire le mapping ancien_id → nouveau_id
    # en tenant compte de gravel → pebble
    mapping = {}
    for cat in d['categories']:
        nom = cat['name']
        if nom == 'gravel':
            nom = 'pebble'  # fusion gravel → pebble
        cible = next((c for c in CATEGORIES_CIBLES if c['name'] == nom), None)
        if cible:
            mapping[cat['id']] = cible['id']

    print(f"\n  Mapping IDs : {mapping}")

    # Réassigner les category_id dans les annotations
    nouvelles_anns = []
    for ann in d['annotations']:
        if ann['category_id'] in mapping:
            ann['category_id'] = mapping[ann['category_id']]
            nouvelles_anns.append(ann)
        # sinon annotation ignorée (catégorie inconnue)

    d['annotations'] = nouvelles_anns
    d['categories']  = CATEGORIES_CIBLES

    # Mettre à jour le chemin images
    d['dossier_images'] = dossier_images

    print(f"  Annotations conservées : {len(nouvelles_anns)}")
    for cat in CATEGORIES_CIBLES:
        n = sum(1 for a in nouvelles_anns if a['category_id'] == cat['id'])
        print(f"    {cat['name']:<15} : {n} annotations")

    return d


#############################
# Normaliser les 3 datasets
#############################
"""Le fonctionnement intrinsèque de normaliser_dataset() est de lire un JSON, de construire un mapping des anciens IDs vers les nouveaux (en tenant compte de la fusion gravel → pebble), puis de réassigner les category_id dans les annotations en utilisant ce mapping. Les annotations dont la catégorie n'est pas dans le mapping sont ignorées (supprimées). Enfin, la fonction met à jour la liste des catégories et ajoute un champ 'dossier_images' pour indiquer où se trouvent les images correspondantes.
"""

ds_original = normaliser_dataset(
    'annotations_clean.json', 'images'
)
ds_set1 = normaliser_dataset(
    'train_set1/train_set1/annotations/instances_default.json',
    'train_set1/train_set1/images'
)
ds_set2 = normaliser_dataset(
    'train_set2/train_set2/annotations/instances_default.json',
    'train_set2/train_set2/images'
)




# ============================================================
# Compilation du dataset final fusionné
# ============================================================



def fusionner_datasets(datasets, dossier_sortie='dataset_fusionne'):
    """
    Fusionne plusieurs datasets COCO normalisés en un seul.
    Réassigne les image_id et annotation_id pour éviter les doublons.
    Copie toutes les images dans dossier_sortie/images/
    """
    import shutil
    os.makedirs(f'{dossier_sortie}/images', exist_ok=True)

    json_fusionne = {
        'licenses'   : [],
        'info'       : {},
        'categories' : CATEGORIES_CIBLES,
        'images'     : [],
        'annotations': []
    }

    image_id_counter = 1
    ann_id_counter   = 1

    for ds in datasets:
        dossier_images = ds['dossier_images']

        # Mapping ancien image_id → nouveau image_id
        mapping_images = {}

        for img in ds['images']:
            # Copier l'image dans le dossier fusionné
            src = os.path.join(dossier_images, img['file_name'])
            dst = os.path.join(dossier_sortie, 'images', img['file_name'])

            if os.path.exists(src):
                shutil.copy2(src, dst)
                mapping_images[img['id']] = image_id_counter

                json_fusionne['images'].append({
                    'id'       : image_id_counter,
                    'file_name': img['file_name'],
                    'width'    : img['width'],
                    'height'   : img['height']
                })
                image_id_counter += 1
            else:
                print(f" Image non trouvée : {src}")

        # Réassigner les annotations
        for ann in ds['annotations']:
            if ann['image_id'] not in mapping_images:
                continue

            json_fusionne['annotations'].append({
                'id'          : ann_id_counter,
                'image_id'    : mapping_images[ann['image_id']],
                'category_id' : ann['category_id'],
                'segmentation': ann['segmentation'],
                'area'        : ann['area'],
                'bbox'        : ann['bbox'],
                'iscrowd'     : ann.get('iscrowd', 0)
            })
            ann_id_counter += 1

    # Sauvegarder
    with open(f'{dossier_sortie}/annotations.json', 'w') as f:
        json.dump(json_fusionne, f)

    print(f"\n Dataset fusionné exporté dans '{dossier_sortie}/'")
    print(f"   Images      : {len(json_fusionne['images'])}")
    print(f"   Annotations : {len(json_fusionne['annotations'])}")

    from collections import Counter
    distrib = Counter(a['category_id'] for a in json_fusionne['annotations'])
    cat_map = {c['id']: c['name'] for c in CATEGORIES_CIBLES}
    print(f"\nDistribution finale :")
    for cat_id, count in sorted(distrib.items()):
        print(f"  {cat_map[cat_id]:<15} : {count:>5} annotations")

# Fusionner
fusionner_datasets(
    [ds_original, ds_set1, ds_set2],
    dossier_sortie='dataset_fusionne'
)


"""
Distribution finale :
  pebble          :   561 annotations
  vegetation      :  1079 annotations
  target          :    45 annotations
"""





# ============================================================
# Augmentation des données des 3 datasets fusionnés
# ============================================================


# Charger le dataset fusionné
with open('dataset_fusionne/annotations.json', 'r') as f:
    data_fusionne = json.load(f)

images_dict_fusionne = {img['id']: img for img in data_fusionne['images']}
categories_fusionne  = {cat['id']: cat['name'] for cat in data_fusionne['categories']}

print("=== AUGMENTATION DU DATASET FUSIONNÉ ===")
print(f"Images source : {len(data_fusionne['images'])}")

# Images contenant du pebble (classe minoritaire)
images_avec_pebble = set(
    a['image_id'] for a in data_fusionne['annotations']
    if a['category_id'] == 1
)
print(f"Images avec pebble : {len(images_avec_pebble)}")
print(f"Images sans pebble : {len(data_fusionne['images']) - len(images_avec_pebble)}")

# Lancer l'augmentation
# n_augmentations=5 par défaut, 15 pour les images avec pebble
exporter_dataset_augmente(
    data_fusionne,
    images_dict_fusionne,
    categories_fusionne,
    transform=transform, #voir la définition du pipeline d'augmentation plus haut sous transform = A.Compose([...])
    n_augmentations=5,
    dossier_images_src='dataset_fusionne/images',
    dossier_sortie='dataset_augmente'   # ← écrase l'ancien
)



# ============================================================
# Split des données en train/test - 80/20
# ============================================================

# imgs (liste d'images) + data_aug (JSON complet)
#               ↓
#   filtrage des annotations correspondantes
#               ↓
#   copie des images sur le disque
#               ↓
#   export d'un JSON COCO autonome
#               ↓
# dataset_train/ ou dataset_val/ prêt à l'emploi



import random
import json
import os
import shutil
from collections import Counter

# Charger le dataset augmenté
with open('dataset_augmente/annotations.json', 'r') as f:
    data_aug = json.load(f)

# Mélanger les images aléatoirement (seed fixe pour reproductibilité)
random.seed(42)
images = data_aug['images'].copy()
random.shuffle(images)

# Split 80/20
split    = int(len(images) * 0.8)
train_imgs = images[:split]
val_imgs   = images[split:]

print(f"Total  : {len(images)} images")
print(f"Train  : {len(train_imgs)} images")
print(f"Val    : {len(val_imgs)} images")

def creer_split(imgs, data_aug, dossier_sortie):
    """Crée un dataset COCO pour un subset d'images"""
    os.makedirs(f'{dossier_sortie}/images', exist_ok=True)

    ids_images = {img['id'] for img in imgs}

    # Filtrer les annotations correspondantes
    annotations = [a for a in data_aug['annotations']
                   if a['image_id'] in ids_images]

    # Copier les images
    for img in imgs:
        src = os.path.join('dataset_augmente/images', img['file_name'])
        dst = os.path.join(dossier_sortie, 'images', img['file_name'])
        if os.path.exists(src):
            shutil.copy2(src, dst)

    # Sauvegarder le JSON
    json_split = {
        'licenses'   : data_aug.get('licenses', []),
        'info'       : data_aug.get('info', {}),
        'categories' : data_aug['categories'],
        'images'     : imgs,
        'annotations': annotations
    }
    with open(f'{dossier_sortie}/annotations.json', 'w') as f:
        json.dump(json_split, f)

    # Stats
    distrib = Counter(a['category_id'] for a in annotations)
    cat_map = {c['id']: c['name'] for c in data_aug['categories']}
    print(f"\n {dossier_sortie}/")
    print(f"   Images      : {len(imgs)}")
    print(f"   Annotations : {len(annotations)}")
    for cat_id, count in sorted(distrib.items()):
        print(f"   {cat_map[cat_id]:<15} : {count:>4}")

creer_split(train_imgs, data_aug, 'dataset_train')
creer_split(val_imgs,   data_aug, 'dataset_val')
