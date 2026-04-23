# ============================================================
# SAM2_inference.py
# Inférence avec le modèle fine-tuné sur de nouvelles images
# ============================================================

import os
import json
import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pycocotools import mask as coco_mask
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ============================================================
# 1. CONFIGURATION
# ============================================================

DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'
POIDS_SAM2   = 'sam2.1_hiera_small.pt'
CONFIG_SAM2  = 'configs/sam2.1/sam2.1_hiera_s.yaml'
POIDS_FINETUNED = 'sam2_riviere_finetuned.pt'
JSON_VAL     = 'dataset_val/annotations.json'
DOSSIER_VAL  = 'dataset_val/images'

CATEGORIES = {1: 'pebble', 2: 'vegetation', 3: 'target'}
COULEURS   = {
    0: (0,   0,   0),    # fond — noir
    1: (255, 140, 0),    # pebble — orange
    2: (0,   200, 50),   # vegetation — vert
    3: (0,   100, 255),  # target — bleu
}

print(f"Device : {DEVICE}")

# ============================================================
# 2. CHARGEMENT DU MODÈLE FINE-TUNÉ
# ============================================================

print("Chargement du modèle fine-tuné...")
sam2_model = build_sam2(CONFIG_SAM2, POIDS_SAM2, device=DEVICE)
predictor  = SAM2ImagePredictor(sam2_model)

# Charger les poids fine-tunés
sam2_model.load_state_dict(
    torch.load(POIDS_FINETUNED, map_location=DEVICE)
)
sam2_model.eval()
print("Modèle fine-tuné chargé")

# ============================================================
# 3. FONCTION D'INFÉRENCE
# ============================================================

def inferer_image(img_path, annotations_image, h, w):
    """
    Prédit les masques pour une image en utilisant les bbox
    des annotations comme prompts.
    Retourne le masque sémantique prédit.
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    predictor.set_image(img)
    masque_pred = np.zeros((h, w), dtype=np.uint8)

    for ann in annotations_image:
        cat_id = ann['category_id']
        bbox   = np.array([
            float(ann['bbox'][0]),
            float(ann['bbox'][1]),
            float(ann['bbox'][0] + ann['bbox'][2]),
            float(ann['bbox'][1] + ann['bbox'][3])
        ])

        # Centroïde depuis bbox
        cx = int((bbox[0] + bbox[2]) / 2)
        cy = int((bbox[1] + bbox[3]) / 2)

        with torch.no_grad():
            sparse_emb, dense_emb = sam2_model.sam_prompt_encoder(
                points = (
                    torch.tensor([[cx, cy]], device=DEVICE,
                                 dtype=torch.float32).unsqueeze(0),
                    torch.tensor([[1]], device=DEVICE, dtype=torch.int)
                ),
                boxes  = torch.tensor(bbox, device=DEVICE,
                                      dtype=torch.float32).unsqueeze(0),
                masks  = None
            )

            low_res_masks, _, _, _ = sam2_model.sam_mask_decoder(
                image_embeddings         = predictor._features['image_embed'],
                image_pe                 = sam2_model.sam_prompt_encoder
                                                     .get_dense_pe(),
                sparse_prompt_embeddings = sparse_emb,
                dense_prompt_embeddings  = dense_emb,
                multimask_output         = False,
                repeat_image             = False,
                high_res_features        = predictor._features.get(
                                               'high_res_feats')
            )

            pred_masks = torch.nn.functional.interpolate(
                low_res_masks,
                size          = (h, w),
                mode          = 'bilinear',
                align_corners = False
            )

        masque_binaire = (torch.sigmoid(pred_masks[0, 0]) > 0.5).cpu().numpy()
        masque_pred[masque_binaire] = cat_id

    return img, masque_pred

# ============================================================
# 4. FONCTION DE VISUALISATION
# ============================================================

def visualiser_comparaison(img, masque_gt, masque_pred, titre):
    """Affiche image originale, masque GT et masque prédit côte à côte"""

    def coloriser_masque(masque):
        couleur = np.zeros((*masque.shape, 3), dtype=np.uint8)
        for cat_id, rgb in COULEURS.items():
            couleur[masque == cat_id] = rgb
        return couleur

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(titre, fontsize=12)

    # Image originale
    axes[0].imshow(img)
    axes[0].set_title('Image originale')
    axes[0].axis('off')

    # Masque ground truth
    axes[1].imshow(coloriser_masque(masque_gt))
    axes[1].set_title('Masque GT (annotations)')
    axes[1].axis('off')

    # Masque prédit
    axes[2].imshow(coloriser_masque(masque_pred))
    axes[2].set_title('Masque prédit (SAM2 fine-tuné)')
    axes[2].axis('off')

    # Légende
    legende = [Patch(color=np.array(COULEURS[i])/255,
                     label=f"{CATEGORIES.get(i, 'fond')}")
               for i in [1, 2, 3]]
    axes[2].legend(handles=legende, loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.show()

    # Stats
    total = masque_pred.size
    print(f"\nCouverture prédite :")
    for cat_id, cat_name in CATEGORIES.items():
        n   = np.sum(masque_pred == cat_id)
        pct = n / total * 100
        print(f"  {cat_name:<15} : {pct:.1f}%")


def calculer_dice_par_classe(masque_pred, masque_gt, categories):
    """
    Calcule le Dice par classe en ignorant les classes
    absentes du masque GT — évite les faux 1.000
    """
    resultats = {}
    for cat_id, cat_name in categories.items():
        pred = (masque_pred == cat_id).astype(float)
        gt   = (masque_gt   == cat_id).astype(float)

        if gt.sum() == 0:
            resultats[cat_name] = None  # classe absente du GT
            continue

        inter = (pred * gt).sum()
        union = pred.sum() + gt.sum()
        dice  = (2 * inter + 1e-6) / (union + 1e-6)
        resultats[cat_name] = round(float(dice), 3)

    return resultats


# ============================================================
# 5. TEST SUR PLUSIEURS IMAGES DU VAL SET
# ============================================================

with open(JSON_VAL, 'r') as f:
    data_val = json.load(f)

images_dict = {img['id']: img for img in data_val['images']}

# Grouper les annotations par image
from collections import defaultdict
anns_par_image = defaultdict(list)
for ann in data_val['annotations']:
    anns_par_image[ann['image_id']].append(ann)

# Tester sur 5 images aléatoires
import random
random.seed(123)
images_test = random.sample(data_val['images'], 5)

for img_info in images_test:
    img_path = os.path.join(DOSSIER_VAL, img_info['file_name'])
    h, w     = img_info['height'], img_info['width']
    anns     = anns_par_image[img_info['id']]

    if not anns or not os.path.exists(img_path):
        continue

    print(f"\nInférence : {img_info['file_name']}")

    # Construire le masque GT
    masque_gt = np.zeros((h, w), dtype=np.uint8)
    for ann in anns:
        rle    = coco_mask.frPyObjects(ann['segmentation'], h, w)
        masque = coco_mask.decode(rle)
        if masque.ndim == 3:
            masque = masque[:, :, 0]
        masque_gt[masque == 1] = ann['category_id']

    # Inférence
    img, masque_pred = inferer_image(img_path, anns, h, w)

    # Calcul Dice par classe
    dices = calculer_dice_par_classe(masque_pred, masque_gt, CATEGORIES)
    print(f"Dice par classe :")
    for cat_name, dice in dices.items():
        if dice is None:
            print(f"  {cat_name:<15} : absent du GT")
        else:
            print(f"  {cat_name:<15} : {dice:.3f}")


"""
G0046895 : pebble=0.368  veg=absent   target=absent
G0046957 : pebble=0.367  veg=0.000    target=0.516
G0174662 : pebble=absent veg=0.740    target=0.983
G0046951 : pebble=0.488  veg=absent   target=0.990
G0178816 : pebble=0.001  veg=0.628    target=0.971


target     → très bon  (0.516 à 0.990) objet petit et distinct
vegetation → moyen     (0.000 à 0.740) variable selon l'image
pebble     → faible    (0.001 à 0.488) classe la plus difficile
C'est cohérent avec ce qu'on attendait :

target est un objet visuellement très distinct → facile à segmenter
pebble a des textures très variées et des formes irrégulières → difficile   """



