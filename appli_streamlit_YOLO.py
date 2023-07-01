# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 14:45:57 2023

@author: s002135

Pour lancer, ouvrir un prompt, se placer dans le répértoire et taper :
streamlit run appli_streamlit.py

"""

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from yolov5.detect_modifie_jmr import detect_modifie_jmr
import cv2

#Librairie pour visualiser les images avec zoom
import plotly.express as px

annotation_selection_help = "Select an annotation file to load"
upload_help = "Upload a file to annotate"
upload_button_text = "Upload"
upload_button_text_desc = "Choose a file"

st.title('Bienvenue dans le monde de la detection de véhicule')
st.title('Un projet réalisé par :\n Charaf, Theo et Jean-Michel')

# Taille des labels
font_cv2 = cv2.FONT_HERSHEY_SIMPLEX # font
org = (00, 185)# org
fontScale = 1# fontScale
color = (255, 0, 0) # Red color in BGR
thickness = 2# Line thickness of 2 px

# Partie chargement d'une image

with st.form("upload-form", clear_on_submit=False):
    uploaded_file = st.file_uploader(upload_button_text_desc, accept_multiple_files=False,
                                     type=['png', 'jpg', 'jpeg'],
                                     help=upload_help)
    submitted = st.form_submit_button(upload_button_text)
    

    if submitted and uploaded_file is not None:
        file1=uploaded_file.name
        st.write('You selected:', file1)
        

# séparation de l'affichage en 2 colonnes
col1, col2 = st.columns(2)

nb_sedan=0
nb_minibus=0
nb_truck=0
nb_pickup=0
nb_bus=0
nb_cement_truck=0
nb_trailer=0
label=[]
xyxy=[]
model_valid=False

# Traitement de la colonne 1 à gauche
with col1:
    option = st.selectbox(
        'Which model do you want to test?',
        ('YOLO', 'Charaf', 'Theo'))

    st.write('You selected:', option)
    
    if option == 'YOLO':
        model_valid=True
        st.write(option, 'implémenté')
        if submitted and uploaded_file is not None:

            img = Image.open(uploaded_file)
            lien=os.path.join(os.getcwd(),"img.jpg")
            img = img.save(lien)
            detect_modifie_jmr(lien,label,xyxy)
            nb_sedan=np.size(label)
    else:
        model_valid=False
        st.write(option, 'non implémenté')    
    
    col11, col12 = st.columns(2)

    with col11:
        st.write('sedan')
        st.write('minibus')
        st.write('truck')
        st.write('pickup')
        st.write('bus')    
        st.write('cement truck')  
        st.write('trailer')  

    with col12:
        st.write(nb_sedan)
        st.write(nb_minibus)
        st.write(nb_truck)
        st.write(nb_pickup)
        st.write(nb_bus)    
        st.write(nb_cement_truck)  
        st.write(nb_trailer)  
    

# Traitement de la colonne 2 à droite
with col2:

    labels = {
            'x':"Axe X",
            'y':"Axe Y" ,
            'color':'Z Label'      
            }

    if submitted and uploaded_file is not None and model_valid:
     
        
        image = Image.open(uploaded_file)
        st.write("Taille de l'image en pixel :",image.size)
        img1 = ImageDraw.Draw(image)
        nb_sedan=22
        
        for i,box in enumerate(xyxy):
            x1=int(box[0][0])
            y1=int(box[0][1])
            x2=int(box[0][2])
            y2=int(box[0][3])    
            start_point = (int(box[0][0]), int(box[0][1]))
            end_point   = (int(box[0][2]), int(box[0][3]))
            size=[start_point,end_point]
            img1.rectangle(size, fill=None, outline='blue', width=2)
            text = str(label[i]) #text
            org=(min(x1,x2),max(y1,y2))
            img1.text(org,text, fill=(255, 0, 0, 128))

        fig = px.imshow(image,aspect='equal',labels = labels)  

        st.plotly_chart(fig)