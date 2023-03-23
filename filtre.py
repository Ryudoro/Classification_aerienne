import cv2
import matplotlib.pyplot as plt
import math
import numpy as np
voiture = cv2.imread('JPEGImages/004053.jpg')

# couleur HSV
hsv = cv2.cvtColor(voiture, cv2.COLOR_BGR2HSV)

# On définit la plage de couleurs pour filtrer les pixels verts
lower_green = np.array([0, 30, 0])
upper_green = np.array([60, 255, 255])

# On applique un masque pour éviter les pixels verts
mask = cv2.inRange(hsv, lower_green, upper_green)
inv_mask = cv2.bitwise_not(mask)

kernel = np.ones((10,10),np.uint8)
inv_mask = cv2.dilate(inv_mask,kernel,iterations = 2)
inv_mask = cv2.erode(inv_mask,kernel,iterations = 2)

result = cv2.bitwise_and(voiture, voiture, mask=inv_mask)


# On cherche les contours dans le masque
contours, _ = cv2.findContours(inv_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Trouver le plus grand contour qui est probablement celui de la route
max_contour = max(contours, key=cv2.contourArea)

# Dessiner le contour de la route sur une image vide
inv_mask = np.zeros_like(inv_mask)
mask_tot = cv2.drawContours(inv_mask, [max_contour], 0, (255, 255, 255), thickness=-1)

result = cv2.bitwise_and(voiture, voiture, mask=inv_mask)

# Afficher l'image originale et l'image résultante sans les nuances de vert
cv2.imshow('Image originale', voiture)
cv2.imshow('Image sans nuances de vert', result)


# Définir la plage de couleurs pour la route grise
lower_gray = np.array([0, 0, 60])
upper_gray = np.array([300, 40, 150])

# Appliquer un masque pour ne garder que les pixels dans la plage de couleurs qu'on a choisi
mask = cv2.inRange(hsv, lower_gray, upper_gray)

# Appliquer une transformation morphologique  pour réduire le bruit (dans le masque)
kernel = np.ones((10,10),np.uint8)
mask = cv2.dilate(mask,kernel,iterations = 1)
mask = cv2.erode(mask,kernel,iterations = 1)


# Trouver les contours dans le masque
contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Trouver le plus grand contour qui est probablement celui de la route (à ajuster si on en veut plusieurs)
max_contour = max(contours, key=cv2.contourArea)

# Dessiner le contour de la route sur une image vide (comme avant quoi)
mask = np.zeros_like(mask)
cv2.drawContours(mask, [max_contour], 0, (255, 255, 255), thickness=-1)

lower_hsv = np.uint8([lower_gray[0]-300, lower_gray[1]/2, lower_gray[2]/2])
upper_hsv = np.uint8([upper_gray[0]+300, upper_gray[1]*2, upper_gray[2]*2])
hsv_mask = cv2.inRange(hsv, lower_hsv, upper_hsv)


result = cv2.bitwise_and(voiture, voiture, mask= mask)

cv2.waitKey(0)