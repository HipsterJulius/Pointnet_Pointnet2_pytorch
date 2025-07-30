import os
import sys
import numpy as np
import torch
import importlib

# 1. Punktwolke laden
points = np.load("/shared/bike_001_noise_add_0_1.npy").astype(np.float32)

# Auf 1024 Punkte bringen
if points.shape[0] > 1024:
    indices = np.random.choice(points.shape[0], 1024, replace=False)
    points = points[indices]
elif points.shape[0] < 1024:
    indices = np.random.choice(points.shape[0], 1024, replace=True)
    points = points[indices]

points = torch.from_numpy(points).unsqueeze(0).transpose(2, 1).cuda()

# 2. Modellname & Speicherort
log_dir = "pointnet2_ssg_wo_normals"
experiment_dir = os.path.join("log/classification", log_dir)
model_name = "pointnet2_cls_ssg"  # direkt gesetzt

# Modellordner zum Python-Pfad hinzufügen (für importlib)
sys.path.append(experiment_dir)

# 3. Modell laden
model = importlib.import_module(model_name)
num_classes = 40  # anpassen, falls nötig

classifier = model.get_model(num_classes, normal_channel=False).cuda()
checkpoint = torch.load(os.path.join(experiment_dir, 'checkpoints', 'best_model.pth'))
classifier.load_state_dict(checkpoint['model_state_dict'])
classifier.eval()

# 4. Inferenz
with torch.no_grad():
    pred, _ = classifier(points)
    pred_choice = pred.data.max(1)[1].item()

print("Vorhersage (Klasse):", pred_choice)

with open("shape_names.txt", "r") as f:
    shape_names = [line.strip() for line in f]
print("Vorhersage:", shape_names[pred_choice])

with open("/shared/prediction.txt", "w") as f:
    f.write(str(pred_choice))
