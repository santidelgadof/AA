import cv2
import numpy as np

# Leemos la imagen que nos pasas
imagen = cv2.imread("Captura1.PNG")

# Convertimos la imagen a espacio de color HSV
hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

# Definimos el rango de color amarillo en HSV
amarillo_bajo = np.array([20, 80, 110])
amarillo_alto = np.array([62, 255, 255])
# Creamos una máscara binaria con solo los píxeles dentro del rango de amarillo
mascara = cv2.inRange(hsv, amarillo_bajo, amarillo_alto)
cv2.imshow("Mascara", mascara)

# Resultado
cv2.imshow("Original", imagen)
cv2.waitKey(0)
cv2.destroyAllWindows()


