# -*- coding: utf-8 -*-
import os
import time
import sys
import numpy as np
import torch
import importlib

# === Konfiguration ===
SHARED_DIR = "/shared"
POINTCLOUD_SRC = os.path.join(SHARED_DIR, "input.npy")
PREDICTION_PATH_DST = os.path.join(SHARED_DIR, "prediction_pointnet2.txt")
DONE_PATH = os.path.join(SHARED_DIR, "done_pointnet2.txt")

LOG_DIR = "pointnet2_ssg_wo_normals"
EXPERIMENT_DIR = os.path.join("/workspace/Pointnet_Pointnet2_pytorch/log/classification", LOG_DIR)
MODEL_NAME = "pointnet2_cls_ssg"
NUM_POINTS = 1024
NUM_CLASSES = 40  # Anpassen je nach CLASS_MAP
NORMAL_CHANNEL = False

# Modell importieren
sys.path.append("/workspace/Pointnet_Pointnet2_pytorch/models")
model_module = importlib.import_module(MODEL_NAME)
model = model_module.get_model(NUM_CLASSES, normal_channel=NORMAL_CHANNEL)

checkpoint_path = os.path.join(EXPERIMENT_DIR, 'checkpoints', 'best_model.pth')
checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("PointNet2-Modell geladen.")

# Sicherstellen, dass keine alte Prediction-Datei existiert
if os.path.exists(PREDICTION_PATH_DST):
    os.remove(PREDICTION_PATH_DST)

# === Evaluationsschleife ===
print("PointNet2 eval runner gestartet...")
while True:
    if os.path.exists(DONE_PATH):
        print("Stoppsignal empfangen.")
        os.remove(DONE_PATH)
        break

    if os.path.exists(POINTCLOUD_SRC):
        try:
            # Punktwolke laden
            points = np.load(POINTCLOUD_SRC).astype(np.float32)

            # Auf NUM_POINTS bringen
            if points.shape[0] > NUM_POINTS:
                idx = np.random.choice(points.shape[0], NUM_POINTS, replace=False)
            else:
                idx = np.random.choice(points.shape[0], NUM_POINTS, replace=True)
            points = points[idx]

            # Torch-Tensor vorbereiten
            pc_tensor = torch.from_numpy(points).unsqueeze(0).transpose(2, 1)  # [1, 3, N]

            with torch.no_grad():
                pred, _ = model(pc_tensor)
                pred_label = torch.argmax(pred, dim=1).item()

            # Vorhersage speichern
            with open(PREDICTION_PATH_DST, "w") as f:
                f.write(str(pred_label))

            # Eingabe löschen
            os.remove(POINTCLOUD_SRC)

        except Exception as e:
            print("Fehler beim Verarbeiten:", e)
    else:
        time.sleep(0.05)
