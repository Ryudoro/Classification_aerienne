import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

data_dir = "classification"
batch_size = 32
image_size = (224, 224)
num_classes = 7
epochs = 100

#Je créer un générateur d'image pour augmenter le nombre d'images
train_generateur = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

#J'applique le générateur sur l'ensemble d'entrainement
train_generator = train_generateur.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

#J'applique le générateur sur l'ensemble de test
validation_generator = train_generateur.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

#Modèle préentrainé spécialisé dans la classification d'image (sans les couches supérieures)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

x = base_model.output

#Je préfère ca à Flatten pour avoir moins d'overfitting
x = GlobalAveragePooling2D()(x)

x = Dense(256, activation='relu')(x)

#Aide pour l'overfitting
x = Dropout(0.5)(x)

predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

#Je fige les couches du modèle importé, car il ne faut pas les mettres à jour pendant l'entrainement
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

#Pour éviter d'attendre 1000 ans et faire de l'overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=20, mode='min', restore_best_weights=True)
lr_plateau = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.8, verbose=2, mode='min')

history = model.fit(
    train_generator,
    epochs=epochs,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=[early_stopping, lr_plateau]
)

model.save('vehicule_classifier.h5')