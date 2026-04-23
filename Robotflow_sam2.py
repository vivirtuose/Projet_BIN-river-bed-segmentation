# Robotflow_sam2



# ============================================================
# SAM2_roboflow.py
# Envoi d'images à l'API Roboflow et sauvegarde des résultats
# ============================================================

import os
import json
from inference_sdk import InferenceHTTPClient

# ============================================================
# CONFIGURATION
# ============================================================

API_KEY     = "cFHyEtp6lPYJckQm4R0Y"
WORKSPACE   = "vivians-workspace-tq2um"
WORKFLOW_ID = "sam2"

DOSSIER_IMAGES  = "robotflow_images"
DOSSIER_SORTIE  = "robotflow_images/roboflow_label"

os.makedirs(DOSSIER_SORTIE, exist_ok=True)

# ============================================================
# CONNEXION
# ============================================================

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY
)

# ============================================================
# ENVOI DES IMAGES
# ============================================================

images = [f for f in os.listdir(DOSSIER_IMAGES)
          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

print(f"{len(images)} images trouvées\n")

for i, nom_image in enumerate(images):
    img_path    = os.path.join(DOSSIER_IMAGES, nom_image)
    nom_base    = os.path.splitext(nom_image)[0]
    json_sortie = os.path.join(DOSSIER_SORTIE, f"{nom_base}.json")

    if os.path.exists(json_sortie):
        print(f"[{i+1}/{len(images)}] --> Déjà traité : {nom_image}")
        continue

    print(f"[{i+1}/{len(images)}] pending {nom_image}", end=' ')

    try:
        result = client.run_workflow(
            workspace_name = WORKSPACE,
            workflow_id    = WORKFLOW_ID,
            images         = {"image": img_path},
            use_cache      = True
        )

        with open(json_sortie, 'w') as f:
            json.dump(result, f, indent=2)


    except Exception as e:
        print(f"Erreur {e}")

print(f"\n Résultats sauvegardés dans '{DOSSIER_SORTIE}/'")
