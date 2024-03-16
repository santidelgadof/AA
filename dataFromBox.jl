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

function getSizeRelation(bounding_boxes::Vector{T}) where T
    sizeRelations = Float64[]
    for box in bounding_boxes
        x1, y1, x2, y2 = box
        relation = (x2-x1)/(y2-y1)
        push!(sizeRelations, relation)
    end
    return sizeRelations

end


function wrapData(bounding_boxes::Vector{T}, hw_relations::Vector{Float64}, densities::Vector{Float64}) where T
    inputData = []
    
    for (i,box) in enumerate(bounding_boxes)
        x1, y1, x2, y2 = box
        boxCenter = (x1+x2)/2, (y1+y2)/2
        data = (boxCenter, hw_relations[i], densities[i])
        push!(inputData, data)
    end
    
    return inputData
end