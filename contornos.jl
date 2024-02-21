using FileIO, Plots, Images, ImageView, TestImages, ColorTypes, Colors

# Leemos la imagen que nos pasas
imagen = load("Captura2.PNG")

# Convertimos la imagen a HSV
hsv = convert(HSV, image) # Usamos el punto para aplicar la función a cada elemento

# Definimos el rango de color amarillo en HSV
amarillo_bajo = HSV(20/360, 100/255, 130/255) # Los valores de HSV en Julia están normalizados
amarillo_alto = HSV(62/360, 255/255, 255/255)

# Creamos una máscara binaria con solo los píxeles dentro del rango de amarillo
mascara = [amarillo_bajo.h <= pixel.h <= amarillo_alto.h && amarillo_bajo.s <= pixel.s <= amarillo_alto.s && amarillo_bajo.v <= pixel.v <= amarillo_alto.v ? HSV(0,0,1) : HSV(0,0,0) for pixel in hsv]

# Resultado
imshow(mascara)