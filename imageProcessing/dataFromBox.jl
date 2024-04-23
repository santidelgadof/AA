using Images

function bounding_box_density(bounding_boxes::Vector{T}, total_area::Int) where T
    # Initialize a vector to store bounding box densities
    box_densities = Float64[]
    
    # Calculate the density of each bounding box
    for bbox in bounding_boxes
        x1, y1, x2, y2 = bbox
        # Calculate the area of the bounding box
        bbox_area = (x2 - x1 + 1) * (y2 - y1 + 1)
        # Calculate the density of the bounding box
        density = bbox_area / total_area
        # Add the density to the vector of bounding box densities
        push!(box_densities, density)
    end
    
    return box_densities
end

function density_inside_bounding_box(bounding_boxes::Vector{T}, densities::Vector{Float64}) where T
    # Initialize a vector to store the densities inside each bounding box
    inside_densities = Float64[]
    
    # Iterate over each bounding box
    for (i, bbox) in enumerate(bounding_boxes)
        # Extract the coordinates of the bounding box
        x1, y1, x2, y2 = bbox
        
        # Extract the density of the bounding box
        density = densities[i]
        
        # Create a mask for the bounding box
        mask = zeros(Bool, size(densities))
        mask[y1:y2, x1:x2] .= true
        
        # Calculate the density inside the bounding box
        inside_density = sum(mask .* densities) / density
        
        # Add the inside density to the vector of inside densities
        push!(inside_densities, inside_density)
    end
    
    return inside_densities
end 

#dentro de cada bounding box, la cantidad de blanco con respecto a todos los pixeles de la bounding box
function density_inside_each_bbox(bounding_boxes::Vector{T}, image) where T
    inside_densities = Float64[]

    for bbox in bounding_boxes
        x1, y1, x2, y2 = bbox
        auxiliar_img = image[y1:y2, x1:x2]
        size_total= (x2-x1)*(y2-y1)
        white_pixels = sum(auxiliar_img)
        density = white_pixels/size_total
        push!(inside_densities, density)

    end
    return inside_densities
end

function density_around_center_of_each_bbox(bounding_boxes::Vector{T}, image) where T
    around_center_densities = Float64[]

    for bbox in bounding_boxes
        x1, y1, x2, y2 = bbox
        
        # Calcular el centro de la bounding box
        center_x = (x1 + x2) ÷ 2
        center_y = (y1 + y2) ÷ 2
        
        # Definir la región alrededor del centro
        bbox_size_x = x2 - x1
        bbox_size_y = y2 - y1

        #println("bbox_size_x: ", bbox_size_x, " bbox_size_y: ", bbox_size_y, "\n")
        
        if bbox_size_x <= 10 || bbox_size_y <= 10
            region_percentage = 1  # Tomar el 20% si la bounding box es menor o igual a 10 en algún lado
        else
            region_percentage = 0.2  # Tomar el 10% por defecto
        end
        
        region_size_x = ceil(Int, region_percentage * bbox_size_x)
        region_size_y = ceil(Int, region_percentage * bbox_size_y)
        
        #println("region_size_x: ", region_size_x, " region_size_y: ", region_size_y, "\n")
        # Calcular las coordenadas del área alrededor del centro
        start_x = max(1, center_x - region_size_x ÷ 2)
        end_x = min(size(image, 2), center_x + region_size_x ÷ 2)
        start_y = max(1, center_y - region_size_y ÷ 2)
        end_y = min(size(image, 1), center_y + region_size_y ÷ 2)


        #println("start_x: ", start_x, " end_x: ", end_x, " start_y: ", start_y, " end_y: ", end_y, "\n")
        
        start_x = ceil(Int, start_x)
        end_x = ceil(Int, end_x)
        start_y = ceil(Int, start_y)
        end_y = ceil(Int, end_y)

        #println("start_x: ", start_x, " end_x: ", end_x, " start_y: ", start_y, " end_y: ", end_y, "\n")
        # Extraer la región de interés
        aux_img = image[start_y:end_y, start_x:end_x]
        
        # Calcular el número total de píxeles en la región
        total_pixels = (end_x - start_x + 1) * (end_y - start_y + 1)
        #println("total_pixels: ", total_pixels, "\n")
        
        #println("aux_img: ", aux_img, "\n")
        # Calcular la cantidad de píxeles blancos en la región

        white_pixels = sum(aux_img)
        #println("white_pixels: ", white_pixels, "\n")
        
        # Calcular la densidad de blanco alrededor del centro

        density = white_pixels / total_pixels
        
        push!(around_center_densities, density)
    end

    return around_center_densities
end

function density_in_specific_regions(bounding_boxes::Vector{T}, image) where T
    specific_regions_diff_densities = Float64[]

    for bbox in bounding_boxes
        x1, y1, x2, y2 = bbox
        
        # Calcular el tamaño de las regiones en el eje x
        region_percentage = 0.15
        
        region_size_x = ceil(Int, region_percentage * (x2 - x1))
        
        # Calcular las coordenadas de las regiones
        region1_start_x = x1 + 1
        region1_end_x = region1_start_x + region_size_x - 1
        
        region2_start_x = x2 - region_size_x + 1
        region2_end_x = x2
        
        # Extraer las regiones de interés
        region1 = image[y1:y2, region1_start_x:region1_end_x]
        region2 = image[y1:y2, region2_start_x:region2_end_x]
        
        # Calcular el número total de píxeles en cada región
        total_pixels_region1 = size(region1, 1) * region_size_x
        total_pixels_region2 = size(region2, 1) * region_size_x
        
        # Calcular la cantidad de píxeles blancos en cada región
        white_pixels_region1 = sum(region1)
        white_pixels_region2 = sum(region2)
        
        # Calcular la densidad de blanco en cada región
        density_region1 = white_pixels_region1 / total_pixels_region1
        density_region2 = white_pixels_region2 / total_pixels_region2

        # Calcular la diferencia de densidades en valor absoluto
        diff_density = abs(density_region1 - density_region2)

        push!(specific_regions_diff_densities, diff_density)
    end

    return specific_regions_diff_densities
end


function getSizeRelation(bounding_boxes::Vector{T}) where T
    sizeRelations = Float64[]
    for box in bounding_boxes
        x1, y1, x2, y2 = box
        relation = (x2-x1)/(y2-y1)
        push!(sizeRelations, relation)
    end
    return sizeRelations

end

function normalize_coordinates(box::Tuple{Int, Int, Int, Int}, img_width::Int, img_height::Int)::Tuple{Float64, Float64, Float64, Float64}
    x1, y1, x2, y2 = box
    norm_x1 = x1 / img_width
    norm_y1 = y1 / img_height
    norm_x2 = x2 / img_width
    norm_y2 = y2 / img_height
    return (norm_x1, norm_y1, norm_x2, norm_y2)
end

function wrapData(path, bounding_boxes::Vector{T}, hw_relations::Vector{Float64}, densities::Vector{Float64}, density_around_center::Vector{Float64}, density_in_each_side::Vector{Float64}) where T
    inputData = []
    
    for (i, box) in enumerate(bounding_boxes)
        norm_box = normalize_coordinates(box, 1920,1080)
        center_x = (box[1]+box[3])/2, (box[2]+box[4])/2
        norm_center_x = (norm_box[1]+norm_box[3])/2, (norm_box[2]+norm_box[4])/2
        data = (path, center_x, hw_relations[i], densities[i], density_around_center[i], density_in_each_side[i], norm_center_x)
        push!(inputData, data)
    end
    
    return inputData
end

function resize_and_save_bounding_boxes(bounding_boxes, img, desired_width, desired_height, output_dir)
    resized_boxes = Tuple{Int, Int, Int, Int}[]
    
    for (idx, bbox) in enumerate(bounding_boxes)
        x, y, width, height = bbox
        
        # Asegurarse de que las coordenadas de la región de interés estén dentro de los límites de la imagen
        x = clamp(x, 1, size(img, 2))
        y = clamp(y, 1, size(img, 1))
        width = min(width, size(img, 2) - x + 1)
        height = min(height, size(img, 1) - y + 1)
        
        # Recortar la región de interés de la imagen original
        region_of_interest = img[y:y+height-1, x:x+width-1]
        
        #println("Region Size Before Resize: ", size(region_of_interest))

        # Verificar si la región de interés es lo suficientemente grande para redimensionar
        if size(region_of_interest, 1) >= desired_height && size(region_of_interest, 2) >= desired_width
            # Redimensionar la región recortada al nuevo tamaño deseado
            resized_region = imresize(region_of_interest, (desired_width, desired_height))
            
            # Guardar la región recortada y redimensionada en el directorio de salida
            output_path = joinpath(output_dir, "bbox_$idx.png")
            save(output_path, resized_region)
            
            # Actualizar las coordenadas de la bounding box redimensionada
            new_width, new_height = size(resized_region, 2), size(resized_region, 1)
            new_x = x
            new_y = y
            
            # Guardar las nuevas coordenadas de la bounding box redimensionada
            push!(resized_boxes, (new_x, new_y, new_width, new_height))
            #println("Bounding Box Coordinates: ", bbox)
            #println("Image Size: ", size(img))
        else
            #println("La región recortada es demasiado pequeña para la redimensión. Descartando la bounding box.")
        end
    end
    
    return resized_boxes
end
