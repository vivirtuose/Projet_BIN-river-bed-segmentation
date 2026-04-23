# Pour chaque image du val set :
#         ↓
#   Charger l'image + annotations GT
#         ↓
#   Inférence SAM2 fine-tuné → masque prédit
#         ↓
#   Calculer couverture par classe (% de pixels)
#         ↓
#   Comparer prédit vs GT
#         ↓
# Exporter un CSV avec toutes les stats
# + Graphiques de synthèse



# ============================================================
# SAM2_stats.py
# Statistiques de couverture sur le dataset de validation
# ============================================================

import os
import json
import numpy as np
import cv2
import torch
import csv
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from collections import defaultdict
from pycocotools import mask as coco_mask
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ============================================================
# 1. CONFIGURATION
# ============================================================

DEVICE          = 'cuda' if torch.cuda.is_available() else 'cpu'
POIDS_SAM2      = 'sam2.1_hiera_small.pt'
CONFIG_SAM2     = 'configs/sam2.1/sam2.1_hiera_s.yaml'
POIDS_FINETUNED = 'sam2_riviere_finetuned.pt'
JSON_VAL        = 'dataset_val/annotations.json'
DOSSIER_VAL     = 'dataset_val/images'
CSV_SORTIE      = 'stats_couverture.csv'

CATEGORIES = {1: 'pebble', 2: 'vegetation', 3: 'target'}

print(f"Device : {DEVICE}")

# ============================================================
# 2. CHARGEMENT DU MODÈLE
# ============================================================

print("Chargement du modèle fine-tuné...")
sam2_model = build_sam2(CONFIG_SAM2, POIDS_SAM2, device=DEVICE)
predictor  = SAM2ImagePredictor(sam2_model)
sam2_model.load_state_dict(
    torch.load(POIDS_FINETUNED, map_location=DEVICE)
)
sam2_model.eval()
print("Modèle chargé\n")

# ============================================================
# 3. FONCTIONS UTILITAIRES
# ============================================================

def inferer_masque(img_path, annotations, h, w):
    """Prédit le masque sémantique pour une image"""
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    predictor.set_image(img)
    masque_pred = np.zeros((h, w), dtype=np.uint8)

    for ann in annotations:
        cat_id = ann['category_id']
        bbox   = np.array([
            float(ann['bbox'][0]),
            float(ann['bbox'][1]),
            float(ann['bbox'][0] + ann['bbox'][2]),
            float(ann['bbox'][1] + ann['bbox'][3])
        ])
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

    return masque_pred

def couverture_par_classe(masque, categories):
    """Calcule le % de pixels par classe"""
    total = masque.size
    return {
        cat_name: round(np.sum(masque == cat_id) / total * 100, 2)
        for cat_id, cat_name in categories.items()
    }

def dice_par_classe(masque_pred, masque_gt, categories):
    """Calcule le Dice par classe — ignore les classes absentes du GT"""
    resultats = {}
    for cat_id, cat_name in categories.items():
        pred = (masque_pred == cat_id).astype(float)
        gt   = (masque_gt   == cat_id).astype(float)
        if gt.sum() == 0:
            resultats[cat_name] = None
            continue
        inter = (pred * gt).sum()
        union = pred.sum() + gt.sum()
        resultats[cat_name] = round(float((2*inter+1e-6)/(union+1e-6)), 3)
    return resultats

# ============================================================
# 4. TRAITEMENT DE TOUTES LES IMAGES VAL
# ============================================================

with open(JSON_VAL, 'r') as f:
    data_val = json.load(f)

images_dict    = {img['id']: img for img in data_val['images']}
anns_par_image = defaultdict(list)
for ann in data_val['annotations']:
    anns_par_image[ann['image_id']].append(ann)

# Résultats
resultats = []

print(f"Traitement de {len(data_val['images'])} images...\n")

for i, img_info in enumerate(data_val['images']):
    img_path = os.path.join(DOSSIER_VAL, img_info['file_name'])
    h, w     = img_info['height'], img_info['width']
    anns     = anns_par_image[img_info['id']]

    if not anns or not os.path.exists(img_path):
        continue

    print(f"[{i+1}/{len(data_val['images'])}] {img_info['file_name']}")

    # Masque GT
    masque_gt = np.zeros((h, w), dtype=np.uint8)
    for ann in anns:
        rle    = coco_mask.frPyObjects(ann['segmentation'], h, w)
        masque = coco_mask.decode(rle)
        if masque.ndim == 3:
            masque = masque[:, :, 0]
        masque_gt[masque == 1] = ann['category_id']

    # Inférence
    masque_pred = inferer_masque(img_path, anns, h, w)

    # Stats prédites
    couv_pred = couverture_par_classe(masque_pred, CATEGORIES)
    couv_gt   = couverture_par_classe(masque_gt,   CATEGORIES)
    dices     = dice_par_classe(masque_pred, masque_gt, CATEGORIES)

    resultats.append({
        'image'              : img_info['file_name'],
        # Couverture prédite
        'pred_pebble_%'      : couv_pred['pebble'],
        'pred_vegetation_%'  : couv_pred['vegetation'],
        'pred_target_%'      : couv_pred['target'],
        'pred_fond_%'        : round(100 - sum(couv_pred.values()), 2),
        # Couverture GT
        'gt_pebble_%'        : couv_gt['pebble'],
        'gt_vegetation_%'    : couv_gt['vegetation'],
        'gt_target_%'        : couv_gt['target'],
        # Dice
        'dice_pebble'        : dices['pebble'],
        'dice_vegetation'    : dices['vegetation'],
        'dice_target'        : dices['target'],
    })

print(f"\n {len(resultats)} images traitées")

# ============================================================
# 5. EXPORT CSV
# ============================================================

with open(CSV_SORTIE, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=resultats[0].keys())
    writer.writeheader()
    writer.writerows(resultats)

print(f" Stats exportées → {CSV_SORTIE}")

# ============================================================
# 6. GRAPHIQUES DE SYNTHÈSE
# ============================================================

# Moyennes globales
def moyenne_sans_none(valeurs):
    v = [x for x in valeurs if x is not None]
    return round(np.mean(v), 3) if v else 0

pred_pebble    = [r['pred_pebble_%']     for r in resultats]
pred_veg       = [r['pred_vegetation_%'] for r in resultats]
pred_target    = [r['pred_target_%']     for r in resultats]
gt_pebble      = [r['gt_pebble_%']       for r in resultats]
gt_veg         = [r['gt_vegetation_%']   for r in resultats]
gt_target      = [r['gt_target_%']       for r in resultats]
dices_pebble   = [r['dice_pebble']       for r in resultats]
dices_veg      = [r['dice_vegetation']   for r in resultats]
dices_target   = [r['dice_target']       for r in resultats]

print(f"\n=== STATISTIQUES GLOBALES ===")
print(f"{'Classe':<15} {'Couv. GT':>10} {'Couv. Pred':>12} {'Dice moyen':>12}")
print(f"{'─'*50}")
for nom, gt, pred, dice in [
    ('pebble',     gt_pebble, pred_pebble, dices_pebble),
    ('vegetation', gt_veg,    pred_veg,    dices_veg),
    ('target',     gt_target, pred_target, dices_target),
]:
    print(f"{nom:<15} {np.mean(gt):>9.1f}% "
          f"{np.mean(pred):>11.1f}% "
          f"{moyenne_sans_none(dice):>12.3f}")

# Graphique 1 : couverture moyenne GT vs Prédit
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Statistiques de couverture — Dataset Val (155 images)',
             fontsize=13)

classes = ['pebble', 'vegetation', 'target']
gt_moy  = [np.mean(gt_pebble), np.mean(gt_veg), np.mean(gt_target)]
pred_moy= [np.mean(pred_pebble), np.mean(pred_veg), np.mean(pred_target)]

x = np.arange(len(classes))
axes[0].bar(x - 0.2, gt_moy,   0.4, label='GT',    color='steelblue')
axes[0].bar(x + 0.2, pred_moy, 0.4, label='Prédit', color='orange')
axes[0].set_xticks(x)
axes[0].set_xticklabels(classes)
axes[0].set_title('Couverture moyenne (%)')
axes[0].set_ylabel('%')
axes[0].legend()
axes[0].grid(axis='y')

# Graphique 2 : Dice moyen par classe
dice_moy = [
    moyenne_sans_none(dices_pebble),
    moyenne_sans_none(dices_veg),
    moyenne_sans_none(dices_target)
]
couleurs = ['orange', 'green', 'steelblue']
bars = axes[1].bar(classes, dice_moy, color=couleurs)
axes[1].set_title('Dice Score moyen par classe')
axes[1].set_ylim(0, 1)
axes[1].axhline(y=0.5, color='red', linestyle='--', label='seuil 0.5')
axes[1].legend()
axes[1].grid(axis='y')
for bar, val in zip(bars, dice_moy):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', fontsize=10)

# Graphique 3 : distribution Dice target (la plus stable)
axes[2].hist(
    [d for d in dices_target if d is not None],
    bins=20, color='steelblue', edgecolor='white'
)
axes[2].set_title('Distribution Dice — target')
axes[2].set_xlabel('Dice Score')
axes[2].set_ylabel('Nombre d\'images')
axes[2].grid(axis='y')

plt.tight_layout()
plt.savefig('stats_couverture.png', dpi=150, bbox_inches='tight')
plt.show()
print(" Graphiques sauvegardés → stats_couverture.png")
