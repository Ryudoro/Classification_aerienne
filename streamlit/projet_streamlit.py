import base64
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import seaborn as sns
import altair as alt
import base64
import webview

import os
import cv2
import util

import plotly.express as px

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog

import matplotlib.pyplot as plt

import torch

import fiftyone as fo

import time

#### INITIALISATION DES PARAMETRES

if 'project_url_od' not in st.session_state:
    st.session_state['project_url_od'] = ""
if 'confidence_threshold' not in st.session_state:
    st.session_state['confidence_threshold'] = "40"
if 'overlap_threshold' not in st.session_state:
    st.session_state['overlap_threshold'] = "30"
if 'include_bbox' not in st.session_state:
    st.session_state['include_bbox'] = "Oui"
if 'show_class_label' not in st.session_state:
    st.session_state['show_class_label'] = 'Oui'
if 'box_type' not in st.session_state:
    st.session_state['box_type'] = None
if 'uploaded_file_od' not in st.session_state:
    st.session_state['uploaded_file_od'] = ""

#### FONCTION QUI CHARGE LES MODELES POUR LES METTRE EN CACHE

@st.cache_resource()
def load_models():
    classes_ = ["Berline", "Mini Bus", "Camionnette", "Remorque", "Bus", "Camion ciment", "Poids lourd"]

    device = 'cpu'

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/retinanet_R_101_FPN_3x.yaml'))
    cfg.MODEL.WEIGHTS = './model_final.pth'
    cfg.MODEL.DEVICE = device
    cfg.MODEL.RETINANET.NMS_THRESH_TEST = overlap_threshold/100
    cfg.TEST.DETECTIONS_PER_IMAGE = 200

    detection = DefaultPredictor(cfg)

    # On instancie le modèle de classifictaion ainsi que les transformations associées
    num_classes = len(classes_)

    classifier = util.Classifier(num_classes)
    classifier.to(device)
    classifier.load_state_dict(torch.load('./vaid_classifier.pth', map_location=torch.device('cpu'))) # map_location=torch.device('cpu') permet de charger un modèle entrainer avec cuda sur un cpu
    classifier.eval()

    # Les transformations des véhicules détectés avnat classification sont et doivent les mêmes que celles utilisées lors de l'entrainement du modèles.
    transformes = util.get_transform_inference()


    model_yolo = torch.hub.load('ultralytics/yolov5', 'custom', path='yolo8.pt', device = device)
    return detection, classifier, model_yolo, transformes, classes_

### FONCTION DE DETECTION ET AFFICHAGE DETECTRON 


def detectron_detect(image_array):
    ### Traitement des sorties du modele Detectron/Resnet
    device = 'cpu'
    debut = time.time()
    predictions = util.prediction(image_array, classifier, detection, transformes, device, confiance=confidence_threshold/100)
    fin = time.time()
    temps_execution = fin - debut

    collected_predictions = []
    for i in range(len(predictions)):
        x0 = float(list(predictions[i].pred_boxes)[0][0])
        x1 = float(list(predictions[i].pred_boxes)[0][2])
        y0 = float(list(predictions[i].pred_boxes)[0][1])
        y1 = float(list(predictions[i].pred_boxes)[0][3])
        class_name = classes_[int(list(predictions[i].pred_classes)[0])]
        confidence_score = float(list(predictions[i].scores)[0])
        box = (x0, x1, y0, y1)
        detected_x = int(x0)
        detected_y = int(y0)
        detected_width = int(x1 - x0)
        detected_height = int(y1 - y0)
        roi_bbox = [detected_y, detected_height, detected_x, detected_width]
        collected_predictions.append({"class":class_name, "confidence":confidence_score,
                                    "x0,x1,y0,y1":[int(x0),int(x1),int(y0),int(y1)],
                                    "Width":detected_width, "Height":detected_height,
                                    "ROI, bbox (y+h,x+w)":roi_bbox,
                                    "bbox area (px)":abs(int(x0-x1))*abs(int(y0-y1))})

        start_point = (int(x0), int(y0))
        end_point = (int(x1), int(y1))

        if show_bbox == 'Oui':
            cv2.rectangle(inferenced_img, start_point, end_point, color=(255,0,0), thickness=int(st.session_state['box_width']))

        if show_class_label == 'Oui':
            cv2.putText(inferenced_img,
                class_name,
                (int(x0), int(y0) - 5),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.4,
                color=(255,0,0),
                thickness=int(st.session_state['text_width'])
                )
    return temps_execution, collected_predictions


### FONCTION DE DETECTION ET AFFICHAGE YOLO 

def yolo_detect(image_array):
    debut = time.time()
    predictions = model_yolo([image_array])
    predictions = [det for det in predictions.pred[0] if det[4] > confidence_threshold/100]
    fin = time.time()
    temps_execution = fin - debut

    collected_predictions = []
    for i in range(len(predictions)):
        x0 = float(predictions[i][0])
        x1 = float(predictions[i][2])
        y0 = float(predictions[i][1])
        y1 = float(predictions[i][3])
        class_name = classes_[int(predictions[i][5])]
        confidence_score = float(predictions[i][4])
        box = (x0, x1, y0, y1)
        detected_x = int(x0)
        detected_y = int(y0)
        detected_width = int(x1 - x0)
        detected_height = int(y1 - y0)
        roi_bbox = [detected_y, detected_height, detected_x, detected_width]
        collected_predictions.append({"class":class_name, "confidence":confidence_score,
                                    "x0,x1,y0,y1":[int(x0),int(x1),int(y0),int(y1)],
                                    "Width":detected_width, "Height":detected_height,
                                    "ROI, bbox (y+h,x+w)":roi_bbox,
                                    "bbox area (px)":abs(int(x0-x1))*abs(int(y0-y1))})

        start_point = (int(x0), int(y0))
        end_point = (int(x1), int(y1))

        if show_bbox == 'Oui':
            cv2.rectangle(inferenced_img, start_point, end_point, color=(0,0,255), thickness=int(st.session_state['box_width']))

        if show_class_label == 'Oui':
            cv2.putText(inferenced_img,
                class_name,
                (int(x0), int(y0) - 5),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.4,
                color=(0,0,255),
                thickness=int(st.session_state['text_width'])
                )
    return temps_execution, collected_predictions


### FONCTION QUI SERA CHARGER A CHAQUE MODIFICATION DE PARAMETRE, ELLE COMPRENDS LA DETECTION ET L'AFFICHAGE


def run_inference(uploaded_img, inferenced_img):

    image_array = cv2.cvtColor(uploaded_img, cv2.COLOR_RGB2BGR)

    
    if show_box_type == None:
        st.write("#### Image brut")
        st.image(uploaded_img)

    if show_box_type == 'Detectron':
        ### Traitement des sorties du modele Detectron/Resnet
        temps_execution,  collected_predictions = detectron_detect(image_array)
        
    if show_box_type == 'Yolo v5':
        ### Traitement du modèle YOLO
        temps_execution,  collected_predictions = yolo_detect(image_array)
                
    if show_box_type == 'All':

        temps_detectron,  detectron_predictions = detectron_detect(image_array)
        temps_yolo,  yolo_predictions = yolo_detect(image_array)





    ### AFFICHAGE DES RESULTATS SOUS LE GRAPHIQUES
        
    if show_box_type is not None and show_box_type is not 'All':
        st.write("### Image avec détections")    
        #st.image(inferenced_img, output_format ="jpeg", use_column_width=True)

        image_with_boxes = cv2.cvtColor(inferenced_img, cv2.COLOR_BGR2RGB)
        _, img_encoded = cv2.imencode(".jpg", image_with_boxes)
    
        img_base64 = img_encoded.tobytes()
    
        st.image(img_base64, use_column_width=True, output_format="jpeg")
        
        

        st.markdown(
        """
        <style>
        img {
            cursor: pointer;
            transition: all .2s ease-in-out;
        }
        img:hover {
            transform: scale(1.5);
        }
        </style>
        """,
        unsafe_allow_html=True,
        )


            
        metrics_tab, results_tab, project_tab = st.tabs(["Résultats", "Details", "Project Info"])

        with results_tab:
            st.dataframe(collected_predictions)

        with metrics_tab:
            st.write('### Résumé')
            df = pd.DataFrame(collected_predictions) 

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(label='Total véhicules :', value=len(df))
            col2.metric(label = 'Moyenne de confiance :', value=f"{int(df['confidence'].mean()*100)}%")
            col3.metric(label='Nombre de classes :', value=len(df['class'].unique()))
            col4.metric(label= 'Temps de détection : ', value = f"{round(temps_execution, 2)}s")

            st.write('#### Nombre de véhicules détectés (par classe)')
            vc = df['class'].value_counts()
            fig, ax = plt.subplots(figsize=(10,8)) 
            bars = ax.bar(vc.index, vc.values, color=sns.color_palette('pastel'))
            ax.set_xticks([])
            ax.set_ylabel('Nombre de véhicules')
            ax.legend(bars, vc.index, title="Classes", loc="upper right")

            st.pyplot(fig)


            st.write('#### Distribution des scores de confiance')
            confidences = df['confidence']

            fig, ax = plt.subplots(figsize=(10, 8))
            sns.histplot(confidences, bins=20, ax=ax, kde=True, color='skyblue')
            ax.set_xlabel("Confiance")
            ax.set_ylabel("Fréquence")
            ax.set_xlim(0, 1)

            st.pyplot(fig)
            


            st.write("#### Dimensions des véhicules")

            plt.figure(figsize=(10, 8))
            ax = sns.scatterplot(data=df, x="Width", y="Height", hue="class", palette="pastel", s=150, alpha=0.7)  
            ax.set_xlabel("Largeur (cm)")
            ax.set_ylabel("Hauteur (cm)")
            ax.legend(title="Class")

            a =  df["Width"].max()
            b =  df["Height"].max()
            valmax = max([a, b])
            ax.set_xlim(0, valmax + 10) 
            ax.set_ylim(0, valmax + 10) 

            st.pyplot(plt)

        with project_tab:
            st.write("### Les modèles ont été entrainé à l'aide du data VAID disponible sur le web")
            st.write("Voici un aperçu des données utilisées pour entraîner le modèle.")
            if st.button('FiftyOne'):
                fo.launch_app()
                
            intro_text = """
            ## Introduction

            L'utilisation de l'intelligence artificielle, notamment du deep learning, a ouvert de nouvelles perspectives dans la détection et la classification d'objets dans des images. Ce rapport se concentre sur un projet de deep learning visant à détecter et classifier des véhicules à partir d'images aériennes. Cette approche est particulièrement utile pour la surveillance du trafic, la gestion des infrastructures de transport et la planification urbaine.

            Nous avons entrepris un travail important de pré-traitement des images aériennes issues de la base de données VAID. Les images brutes présentent des perturbations telles que le flou, le bruit et des perspectives déformées. Nous avons appliqué des techniques de pré-traitement pour optimiser la qualité des données en redressant les perspectives, en supprimant le bruit et en améliorant la netteté des images.

            Pour la détection et la classification des véhicules, nous avons exploré différentes architectures de modèles, en mettant l'accent sur les réseaux de neurones convolutionnels (CNN). Nous avons utilisé des modèles personnalisés basés sur TensorFlow ainsi que des architectures de référence telles que « You Only Look Once » (YOLO).

            L'objectif de ce projet est de fournir une solution efficace pour la détection et la classification des véhicules dans des images aériennes. Les résultats obtenus pourront améliorer la surveillance du trafic, évaluer la densité de véhicules, analyser les schémas de déplacement, détecter les congestions routières et faciliter la planification des infrastructures de transport.

            Dans les sections suivantes, nous présenterons en détail les différentes étapes de notre approche, du pré-traitement des images à l'expérimentation des modèles de deep learning. Nous analyserons également les résultats obtenus et discuterons des perspectives d'amélioration et d'application dans des contextes réels.
            """

            st.markdown(intro_text)







    ### COMPARATIF DES MODELES SI CHOIX 'ALL'
    
    if show_box_type == 'All':
        st.write("### Image avec détections") 
        fig= px.imshow(inferenced_img, aspect='equal')
        fig.update_layout(width=800, height=800)
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        st.plotly_chart(fig)
        #st.image(inferenced_img)

        df_detectron = pd.DataFrame(detectron_predictions) 
        df_yolo = pd.DataFrame(yolo_predictions) 

        col1, col2, col3, col4 = st.columns(4)

        col1.write('### Total véhicules')
        col1.metric(label='Detectron', value=len(df_detectron), delta = len(df_detectron)-len(df_yolo))
        col1.metric(label='Yolo', value=len(df_yolo))   

        col2.write('### Moyenne de confiance')
        col2.metric(label = 'Detectron', value=f"{int(df_detectron['confidence'].mean()*100)}%", delta = f"{int(df_detectron['confidence'].mean()*100)- int(df_yolo['confidence'].mean()*100)}%")
        col2.metric(label = 'Yolo', value=f"{int(df_yolo['confidence'].mean()*100)}%")

        col3.write('### Nombre de classes')
        col3.metric(label='Detectron', value=len(df_detectron['class'].unique()))
        col3.metric(label='Yolo', value=len(df_yolo['class'].unique()))

        col4.write('### Temps de détection')
        col4.metric(label= 'Detectron', value = f"{round(temps_detectron, 2)}s", delta = f"{round(temps_yolo-temps_detectron, 2)}s")
        col4.metric(label= 'Yolo', value = f"{round(temps_yolo, 2)}s")

        st.write('#### Distribution des scores de confiance')
        confidences_yolo = df_yolo['confidence']
        confidences_detectron = df_detectron['confidence']

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.histplot(confidences_yolo, bins=20, ax=ax, kde=True, color='blue', label = 'YOLO', alpha = 0.5)
        sns.histplot(confidences_detectron, bins=20, ax=ax, kde=True, color='red', label = "Detectron", alpha = 0.5)
        ax.set_xlabel("Confiance")
        ax.set_ylabel("Fréquence")
        ax.set_xlim(0, 1)
        ax.legend()

        st.pyplot(fig)

        st.write("#### Dimensions des véhicules")

        plt.figure(figsize=(10, 8))
        ax = sns.scatterplot(data=df_yolo, x="Width", y="Height", color="blue", s=150, alpha=0.5, label = 'YOLO')  
        ax = sns.scatterplot(data=df_detectron, x="Width", y="Height", color="red", s=150, alpha=0.5, label = 'Detectron')  
        ax.set_xlabel("Largeur")
        ax.set_ylabel("Hauteur")
        ax.legend()

        a =  df_detectron["Width"].max()
        b =  df_detectron["Height"].max()
        valmax = max([a, b])
        ax.set_xlim(0, valmax + 10) 
        ax.set_ylim(0, valmax + 10) 

        st.pyplot(plt)


        


### SIDEBAR

with st.sidebar:
    st.write("# 1. Image")
    uploaded_file_od = st.file_uploader("Téléchargement de fichier image",
                                        type=["png", "jpg", "jpeg"],
                                        accept_multiple_files=False)
    
    st.write("# 2. Modèle")
    show_box_type = st.selectbox("Selectionner un modèle:",
                                options=("Detectron", "Yolo v5", "All", None),
                                index=0,
                                key="box_type")


    st.write("# 3. Paramètres")
    confidence_threshold = st.slider("Seuil de confiance (%) : quel est le niveau de confiance minimum acceptable pour afficher une boîte englobante ?", 0, 100, 40, 1)
    overlap_threshold = st.slider("Seuil de chevauchement (%) : quelle est la quantité maximale de chevauchement autorisée entre les cadres de délimitation visibles ?", 0, 100, 30, 1)

    st.write("# 4. Visualisation")    
    col_bbox, col_labels = st.columns(2)
    
    with col_bbox:
        show_bbox = st.radio("Afficher les boîtes englobantes:",
                            options=["Oui", "Non"],
                            index=0,
                            key="include_bbox")

    with col_labels:
        show_class_label = st.radio("Afficher les étiquettes de classe:",
                                    options=["Oui", "Non"],
                                    index=0,
                                    key="show_class_label")

    
    box_width = st.selectbox("Largeur des boîtes englobantes:",
                            options=("1", "2", "3", "4", "5"),
                            index=1,
                            key="box_width")
    
    text_width = st.selectbox("Épaisseur du texte de l'étiquette:",
                            options=("1", "2", "3"),
                            index=0,
                            key="text_width")
             

### PROJET

st.title("🛰️ Détection et classification de véhicules sur images aériennes 🚗✈️")

st.write("""
🌍 Bienvenue à cette application de détection de véhicules sur images aériennes! 
Vous pouvez télécharger une image, et nos modèles (Detectron ou YOLO) 🤖 détecteront et classeront les véhicules présents.
Pour obtenir les meilleurs résultats, veillez à ce que l'image soit de bonne qualité et bien éclairée. 📸
""")

detection, classifier, model_yolo, transformes, classes_ = load_models()

if uploaded_file_od != None:
    image = Image.open(uploaded_file_od)
    uploaded_img = np.array(image)
    inferenced_img = uploaded_img.copy()


    run_inference(uploaded_img, inferenced_img)
