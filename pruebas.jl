include("imageProcessing/contornos.jl")

# si se quiere ver las bounding boxes de una imagen en especifico
# basta con indicar el path de la imagen aquí y runear este archivo

path = "Frames/frames2/frame_24.jpg"
imShowFlag = true

processImage(path, imShowFlag)