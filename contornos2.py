import cv2
import numpy as np
from PIL import Image

# Leemos la imagen
imagen = cv2.imread("frames/frame23_83.png")

# Convertimos la imagen a espacio de color HSV
hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

# Definimos el rango de color amarillo en HSV
"""amarillo_bajo = np.array([30, 125, 150])
amarillo_alto = np.array([30, 255, 255])"""

amarillo_bajo = np.array([20, 100, 155])
amarillo_alto = np.array([62, 255, 255])

# Creamos una máscara binaria con solo los píxeles dentro del rango de amarillo
mascara = cv2.inRange(hsv, amarillo_bajo, amarillo_alto)

# Aplicamos un desenfoque Gaussiano para reducir el ruido
#desenfoque = cv2.GaussianBlur(mascara, (5, 5), 0)

# Realizamos operaciones morfológicas para eliminar el ruido
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
dilatacion = cv2.dilate(mascara, kernel, iterations=1)
erosion = cv2.erode(dilatacion, kernel, iterations=1)
circular_kernel= cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1,1))
erosion2= cv2.erode(erosion, circular_kernel, iterations=1)
dilatacion2 = cv2.dilate(erosion2, circular_kernel, iterations=1)

# Encontramos los contornos
contours, hierarchy = cv2.findContours(dilatacion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filtramos los contornos por área (ajusta min_area y max_area según tu caso)
min_area = 50
max_area = 2000
contours_filtrados = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]

# Dibujamos los contornos filtrados
drawing = np.zeros((imagen.shape[0], imagen.shape[1], 3), dtype=np.uint8)
for cnt in contours_filtrados:
    color = (255, 255, 255)  # Blanco
    cv2.drawContours(drawing, [cnt], -1, color, 2)

# Mostrar las imágenes
cv2.imshow("Original", imagen)
cv2.imshow("Mascara", mascara)
#cv2.imshow("Desenfoque", desenfoque)
cv2.imshow("Erosion", erosion)
#cv2.imshow("Dilatacion", dilatacion)
#cv2.imshow("ersosion2", erosion2)
cv2.imshow("dilatacion2", dilatacion2)
img = Image.fromarray(dilatacion2)
img.save("frames_d/frame23_83_d.png")
cv2.waitKey(0)
cv2.destroyAllWindows()
