# Projet BIN — Segmentation de fonds de rivière

Pipeline complet de segmentation sémantique d'images aquatiques (fonds de rivière) par fine-tuning du modèle SAM2 (Segment Anything Model 2) de Meta.

## Contexte

Ce projet a été développé dans le cadre d'une étude sur la caractérisation des substrats de fonds de rivière à partir d'images sous-marines. L'objectif est de segmenter automatiquement trois classes :
- **pebble** — galets et graviers
- **vegetation** — végétation aquatique
- **target** — cibles de calibration

## Architecture du pipeline

```
Images brutes
    |
    v
Annotation manuelle (CVAT) → COCO JSON RLE
    |
    v
Normalisation des datasets (fusion de 3 sources)
    |
    v
Augmentation (Albumentations : flip, rotation, luminosité, bruit)
    |     Stratégie adaptative : ×15 pour images avec pebble (classe rare)
    |                            ×5 pour les autres
    v
Split train/val 80/20 (620 train / 155 val)
    |
    v
Fine-tuning SAM2.1 Hiera Small
    |     Gel : Image Encoder + Prompt Encoder
    |     Entraîné : Mask Decoder (9.2% des paramètres)
    |     Loss : Dice + BCE (0.5/0.5)
    |     20 epochs, LR 5e-5, ReduceLROnPlateau
    v
Inférence + Évaluation
```

## Structure du repository

```
projet_bin/
├── Projet_BIN.py              # Pipeline de données (RLE → masques → augmentation → export)
├── librairies_projet_BIN.txt  # Documentation détaillée des librairies utilisées
├── requirements.txt           # Dépendances Python
├── README.md                  # Ce fichier
└── data/                      # (non inclus dans le repo)
    ├── annotations_clean.json
    ├── dataset_fusionne/
    ├── dataset_augmente/
    ├── dataset_train/
    └── dataset_val/
```

## Résultats

### Métriques finales (Validation set, epoch 20)

| Métrique | Valeur |
|----------|--------|
| **Val Loss** | 0.32 |
| **Val Dice** | 0.61 |

### Dice Score par classe

| Classe | Dice Score |
|--------|------------|
| target | 0.730 |
| vegetation | 0.484 |
| pebble | 0.465 |

### Diagnostic du modèle

Le modèle présente un **comportement conservateur** :
- Sous-estimation systématique de la couverture des masques prédits (plus petits que les masques GT)
- Bordures des objets grignotées
- Faux négatifs > faux positifs
- Cause probable : features de l'Image Encoder non adaptées au domaine aquatique (gel total pendant l'entraînement)

### Évolution de l'entraînement

- **Train Loss** : 0.50 → 0.09 (amélioration continue)
- **Val Loss** : 0.46 → 0.32 (plateau dès epoch 5)
- **Val Dice** : 0.25 → 0.61 (plateau dès epoch 12)
- **Gap Train/Val** : 0.04 → 0.23 (overfitting léger sur les epochs finales)

## Prérequis

- Python 3.11+
- NVIDIA GPU avec CUDA (testé sur RTX 4060 Laptop, 8GB VRAM, CUDA 12.8)
- ~10 GB d'espace disque pour les datasets augmentés

## Installation

```bash
# Cloner le repository
git clone https://github.com/votre-username/projet_bin.git
cd projet_bin

# Créer un environnement virtuel
python -m venv env_bin
source env_bin/bin/activate  # Linux/Mac
# ou
env_bin\Scripts\activate     # Windows

# Installer les dépendances
pip install -r requirements.txt
```

## Utilisation

Le pipeline complet est contenu dans `Projet_BIN.py`. Chaque section peut être exécutée indépendamment en décommentant les blocs correspondants.

### 1. Préparer les données

Placer les annotations COCO JSON et les images dans le dossier `data/`.

### 2. Normalisation et fusion

```python
# Nettoyer les catégories (fusionner gravel → pebble)
nettoyer_categories('annotations.json', 'annotations_clean.json')

# Normaliser et fusionner les datasets
ds_original = normaliser_dataset('annotations_clean.json', 'images')
fusionner_datasets([ds_original], dossier_sortie='dataset_fusionne')
```

### 3. Augmentation

```python
# Charger le dataset fusionné
with open('dataset_fusionne/annotations.json', 'r') as f:
    data_fusionne = json.load(f)

# Lancer l'augmentation
exporter_dataset_augmente(
    data_fusionne,
    images_dict_fusionne,
    categories_fusionne,
    transform=transform,
    n_augmentations=5,
    dossier_images_src='dataset_fusionne/images',
    dossier_sortie='dataset_augmente'
)
```

### 4. Split train/val

```python
creer_split(train_imgs, data_aug, 'dataset_train')
creer_split(val_imgs, data_aug, 'dataset_val')
```

Le code pour le fine-tuning SAM2 et l'inférence est disponible sur demande (non inclus dans ce repository public pour des raisons de taille).

## Librairies principales

- **pycocotools** : Décodage RLE et manipulation des annotations COCO
- **albumentations** : Augmentation de données (transformations spatiales et photométriques)
- **torch** : Fine-tuning du modèle SAM2
- **sam2** : Architecture SAM2.1 Hiera Small (Meta)
- **opencv-python** : Chargement et sauvegarde des images
- **numpy** : Manipulation des masques

Documentation détaillée des librairies : voir `librairies_projet_BIN.txt`.

## Pistes d'amélioration

1. **Dégeler le Prompt Encoder** : permettre au modèle d'apprendre des représentations de prompts adaptées au domaine aquatique
2. **Augmenter le dataset** : collecter plus d'images avec pebble (classe rare)
3. **Augmentation plus agressive** : ajout de ShiftScaleRotate, ElasticTransform
4. **Early stopping** : arrêter l'entraînement à l'epoch 12 (début d'overfitting)
5. **Test-time augmentation (TTA)** : moyenner les prédictions sur plusieurs augmentations de l'image de test

## Licence

Ce projet a été développé à des fins académiques.

## Contact

Pour toute question : vivian.artuose@example.com

---

**Note** : Les données brutes (images et annotations) ne sont pas incluses dans ce repository pour des raisons de confidentialité. Le code fourni illustre la méthodologie complète du pipeline de segmentation.
