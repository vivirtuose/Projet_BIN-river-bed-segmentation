import torch
print(f"CUDA    : {torch.cuda.is_available()}")
print(f"GPU     : {torch.cuda.get_device_name(0)}")
print(f"VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")


"""1. Charger SAM2 pré-entraîné (poids Facebook)
        ↓
2. Geler les couches profondes (encoder)
   → on n'entraîne que le décodeur + quelques couches finales
   → réduit la VRAM nécessaire de 60%
        ↓
3. Dataset loader → lit dataset_train/ image par image
        ↓
4. Boucle d'entraînement
   → passe chaque image + masque ground truth
   → calcule la loss (Dice + BCE)
   → met à jour les poids
        ↓
5. Validation sur dataset_val/ après chaque epoch
        ↓
6. Sauvegarde du meilleur modèle
"""

# import urllib.request
# print("Téléchargement des poids SAM2 small...")
# urllib.request.urlretrieve(
#     "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
#     "sam2.1_hiera_small.pt"
# )
# print("Poids téléchargés (~185 MB)")


# ============================================================
# SAM2_finetune.py
# Fine-tuning de SAM2 sur dataset_train/ pour segmentation
# automatique de fonds de rivière
# ============================================================

import os
import json
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from pycocotools import mask as coco_mask
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from tqdm import tqdm
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# ============================================================
# 1. CONFIGURATION
# ============================================================

DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'
POIDS_SAM2    = 'sam2.1_hiera_small.pt'
CONFIG_SAM2   = 'configs/sam2.1/sam2.1_hiera_s.yaml'
JSON_TRAIN    = 'dataset_train/annotations.json'
JSON_VAL      = 'dataset_val/annotations.json'
DOSSIER_TRAIN = 'dataset_train/images'
DOSSIER_VAL   = 'dataset_val/images'
POIDS_SORTIE  = 'sam2_riviere_finetuned.pt'

N_EPOCHS          = 20
LR                = 5e-5
N_TRAIN_PAR_EPOCH = 500
N_VAL_PAR_EPOCH   = 50

print(f"Device : {DEVICE}")
print(f"GPU    : {torch.cuda.get_device_name(0)}")

# ============================================================
# 2. DATASET
# ============================================================

class RiviereDataset(Dataset):
    def __init__(self, json_path, dossier_images):
        with open(json_path, 'r') as f:
            data = json.load(f)
        self.images_dict = {img['id']: img for img in data['images']}
        self.annotations = data['annotations']
        self.dossier     = dossier_images
        print(f"Dataset chargé : {len(self.annotations)} annotations")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann      = self.annotations[idx]
        img_info = self.images_dict[ann['image_id']]
        h, w     = img_info['height'], img_info['width']

        img = cv2.imread(os.path.join(self.dossier, img_info['file_name']))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        rle    = coco_mask.frPyObjects(ann['segmentation'], h, w)
        masque = coco_mask.decode(rle)
        if masque.ndim == 3:
            masque = masque[:, :, 0]

        coords = np.where(masque > 0)
        cy = int(np.mean(coords[0])) if len(coords[0]) > 0 else h // 2
        cx = int(np.mean(coords[1])) if len(coords[1]) > 0 else w // 2

        bbox = np.array([
            float(ann['bbox'][0]),
            float(ann['bbox'][1]),
            float(ann['bbox'][0] + ann['bbox'][2]),
            float(ann['bbox'][1] + ann['bbox'][3])
        ])

        return {
            'image'      : img,
            'masque'     : masque.astype(np.float32),
            'point'      : np.array([[cx, cy]], dtype=np.float32),
            'bbox'       : bbox,
            'category_id': ann['category_id'],
        }

# ============================================================
# 3. LOSS FUNCTIONS
# ============================================================

def dice_loss(pred, target, eps=1e-6):
    pred  = torch.sigmoid(pred)
    inter = (pred * target).sum()
    union = pred.sum() + target.sum()
    return 1 - (2 * inter + eps) / (union + eps)

def combined_loss(pred, target):
    bce  = nn.BCEWithLogitsLoss()(pred, target)
    dice = dice_loss(pred, target)
    return 0.5 * bce + 0.5 * dice

def calculer_dice(pred_masque, gt_masque, eps=1e-6):
    pred  = (pred_masque > 0.5).float()
    inter = (pred * gt_masque).sum()
    union = pred.sum() + gt_masque.sum()
    return ((2 * inter + eps) / (union + eps)).item()

# ============================================================
# 4. CHARGEMENT DU MODÈLE
# ============================================================

print("\n⏳ Chargement de SAM2 small...")
sam2_model = build_sam2(CONFIG_SAM2, POIDS_SAM2, device=DEVICE)
predictor  = SAM2ImagePredictor(sam2_model)
print("SAM2 chargé")

# Geler tout sauf le mask decoder
for name, param in sam2_model.named_parameters():
    param.requires_grad = 'mask_decoder' in name

params_entraines = sum(p.numel() for p in sam2_model.parameters()
                       if p.requires_grad)
params_total     = sum(p.numel() for p in sam2_model.parameters())
print(f"Paramètres entraînés : {params_entraines:,} / {params_total:,} "
      f"({params_entraines/params_total*100:.1f}%)")

# ============================================================
# 5. OPTIMISEUR ET DATASETS
# ============================================================

optimizer = optim.AdamW(
    [p for p in sam2_model.parameters() if p.requires_grad],
    lr=LR, weight_decay=1e-4
)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2
)

dataset_train = RiviereDataset(JSON_TRAIN, DOSSIER_TRAIN)
dataset_val   = RiviereDataset(JSON_VAL,   DOSSIER_VAL)

random.seed(42)
indices_train = random.sample(range(len(dataset_train)), N_TRAIN_PAR_EPOCH)
indices_val   = random.sample(range(len(dataset_val)),   N_VAL_PAR_EPOCH)

# ============================================================
# 6. FONCTION traiter_batch
# ============================================================

def traiter_batch(sample, predictor, sam2_model, training=True):
    """Accède directement aux logits internes → gradients OK"""
    img   = sample['image']
    gt    = sample['masque']
    point = sample['point']
    bbox  = sample['bbox']

    with torch.autograd.set_grad_enabled(training):
        predictor.set_image(img)

        sparse_emb, dense_emb = sam2_model.sam_prompt_encoder(
            points = (
                torch.tensor(point, device=DEVICE,
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

        orig_h, orig_w = predictor._orig_hw[-1]
        pred_masks = torch.nn.functional.interpolate(
            low_res_masks,
            size        = (orig_h, orig_w),
            mode        = 'bilinear',
            align_corners = False
        )
        pred_tensor = pred_masks[0, 0]
        gt_tensor   = torch.tensor(gt, dtype=torch.float32, device=DEVICE)

        loss = combined_loss(pred_tensor, gt_tensor)
        dice = calculer_dice(pred_tensor, gt_tensor)

    return loss, dice

# ============================================================
# 7. BOUCLE D'ENTRAÎNEMENT
# ============================================================

historique         = {'train_loss': [], 'val_loss': [], 'val_dice': []}
meilleure_val_loss = float('inf')

print(f"\n{'='*50}")
print(f"Début entraînement — {N_EPOCHS} epochs")
print(f"Train : {N_TRAIN_PAR_EPOCH} samples/epoch")
print(f"Val   : {N_VAL_PAR_EPOCH} samples/epoch")
print(f"{'='*50}\n")

for epoch in range(N_EPOCHS):

    # --- Train ---
    sam2_model.train()
    train_losses, train_dices = [], []

    for idx in tqdm(indices_train,
                    desc=f"Epoch {epoch+1}/{N_EPOCHS} [Train]"):
        sample = dataset_train[idx]
        optimizer.zero_grad()
        try:
            loss, dice = traiter_batch(sample, predictor, sam2_model,
                                       training=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                sam2_model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())
            train_dices.append(dice)
        except Exception as e:
            print(f"\n Sample {idx} ignoré : {e}")
            continue

    # --- Val ---
    sam2_model.eval()
    val_losses, val_dices = [], []

    for idx in tqdm(indices_val,
                    desc=f"Epoch {epoch+1}/{N_EPOCHS} [Val]  "):
        sample = dataset_val[idx]
        try:
            loss, dice = traiter_batch(sample, predictor, sam2_model,
                                       training=False)
            val_losses.append(loss.item())
            val_dices.append(dice)
        except Exception as e:
            continue

    train_loss = np.mean(train_losses) if train_losses else 0
    train_dice = np.mean(train_dices)  if train_dices  else 0
    val_loss   = np.mean(val_losses)   if val_losses   else 0
    val_dice   = np.mean(val_dices)    if val_dices    else 0

    historique['train_loss'].append(train_loss)
    historique['val_loss'].append(val_loss)
    historique['val_dice'].append(val_dice)

    scheduler.step(val_loss)

    print(f"\nEpoch {epoch+1}/{N_EPOCHS} — "
          f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.3f} | "
          f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.3f}")

    if val_loss < meilleure_val_loss:
        meilleure_val_loss = val_loss
        torch.save(sam2_model.state_dict(), POIDS_SORTIE)
        print(f" Meilleur modèle sauvegardé → {POIDS_SORTIE}")

# ============================================================
# 8. COURBES D'ENTRAÎNEMENT
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(historique['train_loss'], label='Train Loss')
axes[0].plot(historique['val_loss'],   label='Val Loss')
axes[0].set_title('Loss')
axes[0].set_xlabel('Epoch')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(historique['val_dice'], label='Val Dice', color='green')
axes[1].set_title('Dice Score (validation)')
axes[1].set_xlabel('Epoch')
axes[1].set_ylim(0, 1)
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('courbes_entrainement.png')
plt.show()
print("Courbes sauvegardées → courbes_entrainement.png")



"""TEST 1 resultat:
Les courbes montrent un entraînement qui fonctionne mais avec des limitations claires :
Train Loss : 0.63 → 0.29  descend bien — le modèle apprend
Val Loss   : 0.73 → 0.54  descend aussi — pas d'overfitting
Val Dice   : 0.12 → 0.28  progresse mais reste faible

Diagnostic
Le gap entre Train Loss et Val Loss se creuse — c'est du sous-apprentissage plutôt que de l'overfitting :
Sous-apprentissage (notre cas) :
  → le modèle n'a pas assez vu de données pour généraliser
  → 100 samples/epoch × 10 epochs = 1000 passages seulement
  → sur 1394 annotations disponibles

Overfitting (cas inverse) :
  → Train Loss très bas, Val Loss remonte
  → le modèle mémorise sans généraliser

Solutions par ordre de priorité
1. Augmenter N_TRAIN_PAR_EPOCH : 100 → 500
   → exposer le modèle à plus de données par epoch

2. Augmenter N_EPOCHS : 10 → 20
   → donner plus de temps d'apprentissage

3. Augmenter LR légèrement : 1e-5 → 5e-5
   → le modèle converge trop lentement
"""



"""TEST 2 resultat:
Train Loss : 0.50 → 0.09  descend régulièrement
Val Loss   : 0.46 → 0.32  descend mais stagne après epoch 5
Val Dice   : 0.28 → 0.61  bonne progression

Diagnostic
Le gap Train/Val Loss se creuse progressivement :
Epoch 1  : Train=0.50, Val=0.46  → gap de 0.04
Epoch 20 : Train=0.09, Val=0.32  → gap de 0.23
C'est du début d'overfitting — le modèle apprend très bien les 500 samples d'entraînement mais commence à les mémoriser plutôt que généraliser.
Le Val Dice plafonne à ~0.60 depuis epoch 12 — le modèle a atteint sa limite avec ce dataset et ces paramètres.

Interprétation globale
Val Dice 0.61 = le modèle segmente correctement
                61% des pixels au bon endroit
                → correct pour un premier fine-tuning
                → suffisant pour tester l'inférence

Pour aller au-delà de 0.70 ??   
  → plus de données annotées
  → entraîner aussi le prompt encoder
  → data augmentation plus agressive
"""


