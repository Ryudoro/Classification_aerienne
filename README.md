# Classification_aerienne

#L'utilisation du script "reperage_2.py" nécessite deux dossiers contenant les bases de données (data et labels)
#disponible à l'adresse:
#https://drive.google.com/drive/folders/1-I3qeZNdOT295CjSM2-6UFteI0SjL_pW?usp=sharing

#La fonction permet de créer un nouveau dossier classification contenant 7 sous-dossier
#séparant les différents types de véhicules.

# !! Attention !! dans le dossier 1, il y aura un fichier 004148_1.jpg non reconnu, il faut l'enlever.

#Une fois le repertoire créé, il suffit d'utiliser le script test_DL_2.py afin de créer un modèle de deeplearning
#stocké dans un fichier "vehicule_classifier.h5"


#Pour ce qui est du fichier filtre.py, il est pour l'instant prévu uniquement pour des images dans un sous-dossier présent dans le même dossier.
#Il est optimisé pour une seule route (sauf si elles sont reliées)

## Detectron2 + resnet50

Étape 1 : Effectuer un prétraitement pour préparer les données pour chaque modèle.

- "preprocessing_resnet_classification.py" va créer un dossier avec toutes les détections du dataset et créer son csv pour les annotations.
- "preprocessing_detectron2.py" va également créer un dossier pour préparer les données à l'entrainement du modèle en effectuant au passage un split
"train/val"

Etape 2 : Entraienement des modèles.

- "train_detectron2.py" est l'entrainement du modèle de detection. (J'ai jonglé entre Mac et Windows, j'ai donc laissé le choix de ce paramètre en brut)
- "train_classification.ipynb" est un notebook d'entrainement du modèle classification.

Je n'ai pas encore optimisé le code, si vous voulez effectuer des changements sur les transformations de données avant l'apprentissage, il faut se rendre dans le fichier "util.py"
c'est très important, car ces transformations vont également être utilisées lors de la détection des véhicules sur Detectron2.

Etape 3 : Prediction.

Le fichier "prediction_notebook.ipynb" permet de réaliser les prédictions en utilisant les deux modèles de la façon suivant :
Le modèle Detectron2 va détecter n'importe quel véhicule, le modèle resnet50 utilise les prédictions pour classifier les véhicules,
enfin, il remplace les préductions du modèles detectron2 par les prédictions classifiées.
