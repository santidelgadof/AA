using Images, ImageFiltering, FileIO, ImageView, ImageMorphology, ColorTypes, ImageDraw, Colors
include("boxCreation.jl")
include("dataFromBox.jl")
img = load("Frames/edition1/frame23_194.png")
img_hsv = HSV.(img)
yellow_low = HSV{Float32}(40, 100/255, 155/255)
yellow_high = HSV{Float32}(71, 255/255, 255/255)

# Sacar mascara de amarillos
mask = umbralizeYellow(img_hsv, yellow_low, yellow_high)
mask_gray = Gray.(mask)

se = centered(Bool[
    0 0 1 1 1 0 0
    0 1 1 1 1 1 0
    1 1 1 1 1 1 1
    1 1 1 1 1 1 1
    1 1 1 1 1 1 1
    0 1 1 1 1 1 0
    0 0 1 1 1 0 0
])

se2 = centered(Bool[
    0 0 1 0 0  
    0 1 1 1 0  
    1 1 1 1 1  
    0 1 1 1 0 
    0 0 1 0 0 
])

# operacion morfologica cierre
img_erosion1 = closing(mask_gray,se2)

# crea bounding boxes
bbox = bounding_boxes(img_erosion1)

# elimina bounding boxes que no cumplen un minimo de tamaño
filtered_boxes = filterSmallBoxes(bbox, 5)

threshold = 100  # Por ejemplo
threshold2 = 100

# Fusionar las bounding boxes cercanas según distancia entre centros
merged_boxes = merge_close_bounding_boxes(filtered_boxes, threshold)

# dibuja bounding boxes en la imagen
# img_bound = draw_bounding_boxes(merged_boxes, img_erosion1, 1)

#fusionar bounding boxes cercanas según cercanía de bordes
joined_boxes = join_near_bounding_boxes(merged_boxes, img_erosion1, 0.1, 1)


# obtener densidad pixeles blancos
density_each_bbox= density_inside_each_bbox(joined_boxes, img_erosion1)

# obtener relacion alto/ancho
sizeRelations = getSizeRelation(joined_boxes)

# agrupar datos en formato -> (centro de bounding box, relacion ancho/alto, densidad de pixeles blancos)
data = wrapData(joined_boxes, sizeRelations, density_each_bbox)

# Nombre del archivo
filename = "datos.txt"

# Escribir densidades en el archivo
open(filename, "w") do file
    for density in density_each_bbox
        # Escribir la densidad en el archivo seguida de un salto de línea
        write(file, "$density\n")
    end
end


print_bounding_boxes(filtered_boxes)

println("\nMERGED")
print_bounding_boxes(merged_boxes)

println("\nJOINED")
print_bounding_boxes(joined_boxes)
println("\nDENSIDAD BLANCO")
println(density_each_bbox)
println("\nRELACION HEIGHTWIDTH")
println(sizeRelations)
println("\nDATA")
println(data)
println("\nEND")
