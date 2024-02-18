import cv2
import numpy as np

# Leemos la imagen que nos pasas
imagen = cv2.imread("Foto_contorno3.jpg")

# Convertimos la imagen a espacio de color HSV
hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

# Definimos el rango de color amarillo en HSV
amarillo_bajo = np.array([30, 100, 130])
amarillo_alto = np.array([30, 255, 255])

# Creamos una máscara binaria con solo los píxeles dentro del rango de amarillo
mascara = cv2.inRange(hsv, amarillo_bajo, amarillo_alto)

# Aplicamos un umbral binario a la máscara con valor 125
_, umbralizada = cv2.threshold(mascara, 80, 255, cv2.THRESH_BINARY)

# Encontramos los contornos en la imagen umbralizada
contornos, _ = cv2.findContours(umbralizada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Dibujamos los contornos sobre la imagen original
cv2.drawContours(imagen, contornos, -1, (0, 0, 255), 2)

# Contamos el número de contornos encontrados
num_contornos = len(contornos)

# Si el número de contornos es mayor que cero, significa que hay enemigos (Se modificará en un futuro)
if num_contornos > 0:
    print(f"Se detectaron {num_contornos} contornos amarillos")
    print(f"Hay enemigos en la imagen")
else:
    print("No hay enemigos en la imagen")

# Resultado
cv2.imshow("Resultado", imagen)
cv2.waitKey(0)
cv2.destroyAllWindows()


