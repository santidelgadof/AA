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
        
        if bbox_size_x <= 10 || bbox_size_y <= 10
            region_percentage = 1  # Tomar el 20% si la bounding box es menor o igual a 10 en algún lado
        else
            region_percentage = 0.2  # Tomar el 10% por defecto
        end
        
        region_size_x = ceil(Int, region_percentage * bbox_size_x)
        region_size_y = ceil(Int, region_percentage * bbox_size_y)
        
        # Calcular las coordenadas del área alrededor del centro
        start_x = max(1, center_x - region_size_x ÷ 2)
        end_x = min(size(image, 2), center_x + region_size_x ÷ 2)
        start_y = max(1, center_y - region_size_y ÷ 2)
        end_y = min(size(image, 1), center_y + region_size_y ÷ 2)
        
        start_x = ceil(Int, start_x)
        end_x = ceil(Int, end_x)
        start_y = ceil(Int, start_y)
        end_y = ceil(Int, end_y)
        # Extraer la región de interés
        aux_img = image[start_y:end_y, start_x:end_x]
        
        # Calcular el número total de píxeles en la región
        total_pixels = (end_x - start_x) * (end_y - start_y)
        
        # Calcular la cantidad de píxeles blancos en la región
        white_pixels = sum(aux_img)
        
        # Calcular la densidad de blanco alrededor del centro

        density = white_pixels / total_pixels
        
        push!(around_center_densities, density)
    end

    return around_center_densities
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


function wrapData(path,bounding_boxes::Vector{T}, hw_relations::Vector{Float64}, densities::Vector{Float64},density_around_center::Vector{Float64}) where T
    inputData = []
    
    for (i,box) in enumerate(bounding_boxes)
        x1, y1, x2, y2 = box
        boxCenter = (x1+x2)/2, (y1+y2)/2
        data = (path, boxCenter, hw_relations[i], densities[i],density_around_center[i])
        push!(inputData, data)
    end
    
    return inputData
end