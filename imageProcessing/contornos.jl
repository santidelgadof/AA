include("boxCreation.jl")
include("dataFromBox.jl")
include("../librerias.jl")

function processImage(path, imShowFlag)
    img = load(path)
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

    # Fusionar las bounding boxes cercanas según distancia entre centros
    merged_boxes = merge_close_bounding_boxes(filtered_boxes, threshold)

    # dibuja bounding boxes en la imagen
    # img_bound = draw_bounding_boxes(merged_boxes, img_erosion1, 1)

    #fusionar bounding boxes cercanas según cercanía de bordes
    joined_boxes = join_near_bounding_boxes(merged_boxes, img_erosion1, 0.1, 1)

    if imShowFlag
        bboxes_image = draw_bounding_boxes(joined_boxes, img_erosion1, 1)
        display(bboxes_image)
    end

    # obtener densidad pixeles blancos
    densities= density_inside_each_bbox(joined_boxes, img_erosion1)

    # obtener relacion alto/ancho
    sizeRelations = getSizeRelation(joined_boxes)

    #obtener dentro de zona central
    density_around_center= density_around_center_of_each_bbox(joined_boxes, img_erosion1)

    #print_bounding_boxes(filtered_boxes)

    #println("\nMERGED")
    #print_bounding_boxes(merged_boxes)

    #println("\nJOINED")
    #print_bounding_boxes(joined_boxes)


    #println("DENSIDAD BLANCO")
    #println(densities)
    #println("RELACION HEIGHTWIDTH")
    #println(sizeRelations)


    # agrupar datos en formato -> (centro de bounding box, relacion ancho/alto, densidad de pixeles blancos)
    return wrapData(path, joined_boxes, sizeRelations, densities, density_around_center)
    

end



