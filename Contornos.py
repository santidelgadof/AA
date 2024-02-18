# Importamos las librerías necesarias
import cv2
import numpy as np

# Leemos la imagen que nos pasas
imagen = cv2.imread("Foto_contorno.jpg")

# Convertimos la imagen a espacio de color HSV
hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

# Definimos el rango de color amarillo en HSV
amarillo_bajo = np.array([30, 125, 150])
amarillo_alto = np.array([30, 255, 255])

# Creamos una máscara con solo los pixeles dentro del rango de amarillo
mascara = cv2.inRange(hsv, amarillo_bajo, amarillo_alto)

# Encontramos los contornos en la máscara
contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Dibujamos los contornos sobre la imagen original
cv2.drawContours(imagen, contornos, -1, (0, 0, 255), 2)

# Mostramos el resultado
cv2.imshow("Resultado", imagen)
cv2.waitKey(0)
cv2.destroyAllWindows()
