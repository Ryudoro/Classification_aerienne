import os
import shutil
import numpy as np
import torch
from IPython.display import Image
import subprocess

# # Les chemins vers vos dossiers d'images et d'étiquettes
# image_dir = 'data'
# label_dir = 'labels'

# # Les chemins vers les nouveaux dossiers d'entraînement et de validation
# train_image_dir = 'image_train'
# train_label_dir = 'label_train'
# val_image_dir = 'image_valid'
# val_label_dir = 'train_valid'

# # Créer les dossiers s'ils n'existent pas
# os.makedirs(train_image_dir, exist_ok=True)
# os.makedirs(train_label_dir, exist_ok=True)
# os.makedirs(val_image_dir, exist_ok=True)
# os.makedirs(val_label_dir, exist_ok=True)

# # Obtenir la liste de tous les fichiers
# all_files = os.listdir(image_dir)

# # Mélanger la liste de fichiers
# np.random.shuffle(all_files)

# # 80% des données pour l'entraînement, 20% pour la validation
# num_train = int(len(all_files) * 0.8)
# train_files = all_files[:num_train]
# val_files = all_files[num_train:]

# # Déplacer les fichiers dans les nouveaux dossiers
# for file in train_files:
#     shutil.move(os.path.join(image_dir, file), os.path.join(train_image_dir, file))
#     shutil.move(os.path.join(label_dir, os.path.splitext(file)[0] + '.xml'), os.path.join(train_label_dir, os.path.splitext(file)[0] + '.xml'))

# for file in val_files:
#     shutil.move(os.path.join(image_dir, file), os.path.join(val_image_dir, file))
#     shutil.move(os.path.join(label_dir, os.path.splitext(file)[0] + '.xml'), os.path.join(val_label_dir, os.path.splitext(file)[0] + '.xml'))

import os
import xml.etree.ElementTree as ET

# Convert function
def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return [x, y, w, h]

def convert_annotation(image_folder, label_folder, image_file):
    in_file = open(label_folder + '/' + os.path.splitext(image_file)[0] + '.xml')
    out_file = open(image_folder + '/' + os.path.splitext(image_file)[0] + '.txt', 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

classes = ["1"]  # Add your classes here
label_folders = ["label_train", "label_valid"]
image_folders = ["image_train", "image_valid"]

for label_folder, image_folder in zip(label_folders, image_folders):
    image_files = os.listdir(image_folder)
    for image_file in image_files:
        if image_file.endswith('.jpg'): # or .png, .jpeg, etc.
            convert_annotation(image_folder, label_folder, image_file)
            
# Configuration
print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")

# Importer le modèle
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# # Enlever les couches supérieures du modèle
# model.model[-1] = torch.nn.Sequential()

# Enregistrer le modèle modifié
torch.save(model, 'yolov5s_modified.pt')

# Entraîner le modèle modifié sur vos données
subprocess.run(['python', 'yolov5/train.py', '--img', '640', '--batch', '16', '--epochs', '3', 
                '--data', 'data.yaml', '--cfg', 'yolov5/models/yolov5s.yaml', 
                '--weights', 'yolov5s.pt', '--name', 'yolov5s_results'])

# Afficher les résultats
Image(filename='yolov5/runs/train/yolov5s_results/test_batch0_pred.jpg', width=600)  # montrer une image d'entraînement avec des prédictions
Image(filename='yolov5/runs/train/yolov5s_results/test_batch0_labels.jpg', width=600)  # montrer une image d'entraînement avec des vérités de terrain