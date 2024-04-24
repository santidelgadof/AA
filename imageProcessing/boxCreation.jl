


function umbralizeYellow(img_hsv, yellow_low, yellow_high)
    mask = map(img_hsv) do pixel
        h, s, v = pixel.h, pixel.s, pixel.v
        h_low, s_low, v_low = yellow_low.h, yellow_low.s, yellow_low.v
        h_high, s_high, v_high = yellow_high.h, yellow_high.s, yellow_high.v
    
        h_in_range = h_low <= h <= h_high
        s_in_range = s_low <= s <= s_high
        v_in_range = v_low <= v <= v_high
    
        return h_in_range && s_in_range && v_in_range ? 1 : 0
    end
    return mask
end





function bounding_boxes(mask::Array{T, 2}) where {T}
    # Encontrar los objetos en la máscara binaria
    objects = label_components(mask)
    
    # Inicializar un vector para almacenar los bounding boxes de cada objeto
    bounding_boxes_list = Tuple[]
    
    # Iterar sobre cada objeto y calcular su bounding box
    for obj in 1:maximum(objects)
        # Crear una máscara para el objeto actual
        obj_mask = objects .== obj
        
        # Encontrar los índices de los píxeles no negros en la máscara del objeto
        non_black_pixels = findall(!isequal(zero(T)), obj_mask)
        
        # Calcular coordenadas mínimas y máximas de los píxeles no negros
        min_x = minimum(i[2] for i in non_black_pixels)
        max_x = maximum(i[2] for i in non_black_pixels)
        min_y = minimum(i[1] for i in non_black_pixels)
        max_y = maximum(i[1] for i in non_black_pixels)
        
        # Agregar el bounding box a la lista
        push!(bounding_boxes_list, (min_x, min_y, max_x, max_y))
    end
    
    # Devolver la lista de bounding boxes
    return bounding_boxes_list
end


function filterSmallBoxes(boxes::Vector{T}, threshold::Real) where T
    copy = deepcopy(boxes)
    counter = 0
    for i in 1:length(boxes)
        x1, y1, x2, y2 = boxes[i]
        if (x2-x1<threshold || y2-y1<threshold)
            deleteat!(copy, i-counter)
            counter += 1
        end
    end
    return copy
end



function merge_close_bounding_boxes(bounding_boxes_list::Vector{T}, threshold::Real) where T<:Tuple
    # Create a copy of the original list of bounding boxes to avoid modifying it directly
    merged_boxes = deepcopy(bounding_boxes_list)
    
    # Flag to indicate if merges have been performed
    merged = true
    
    # Iterate until no more merges are performed
    while merged
        merged = false
        
        # Iterate over each pair of bounding boxes
        for i in 1:length(merged_boxes)-1
            for j in i+1:length(merged_boxes)
                box1 = merged_boxes[i]
                box2 = merged_boxes[j]
                
                # Calculate the distance between the centers of the bounding boxes
                center1 = ((box1[1] + box1[3]) / 2, (box1[2] + box1[4]) / 2)
                center2 = ((box2[1] + box2[3]) / 2, (box2[2] + box2[4]) / 2)
                distance = sqrt((center1[1] - center2[1])^2 + (center1[2] - center2[2])^2)
                
                # If the distance is less than the threshold, merge the bounding boxes
                if distance < threshold
                    merged_boxes[i] = (min(box1[1], box2[1]), min(box1[2], box2[2]), max(box1[3], box2[3]), max(box1[4], box2[4]))
                    deleteat!(merged_boxes, j)
                    merged = true
                    break
                end
            end
            
            if merged
                break
            end
        end
    end
    
    return merged_boxes
end


function draw_bounding_boxes(bounding_boxes_list::Vector{T}, img, color) where T<:Tuple
    # Hacemos una copia profunda de la imagen para no modificar la original
    img2 = deepcopy(img)
    
    # Iteramos sobre todas las bounding boxes en la lista
    for i in 1:length(bounding_boxes_list)
        # Extraemos las coordenadas de la bounding box actual
        x1, y1, x2, y2 = bounding_boxes_list[i]
        
        # Dibujamos los bordes horizontales de la bounding box
        for x in x1:x2
            img2[y1, x] = color
            img2[y2, x] = color
        end
        
        # Dibujamos los bordes verticales de la bounding box
        for y in y1:y2
            img2[y, x1] = color
            img2[y, x2] = color
        end
    end
    
    # Devolvemos la imagen con las bounding boxes dibujadas
    return img2
end



function remove_nested_boxes(bounding_boxes::Vector{T}) where T<:Tuple
    # Creamos una lista para almacenar las bounding boxes que no están dentro de otras
    non_nested_boxes = Vector{T}()

    # Iteramos sobre todas las bounding boxes
    for box1 in bounding_boxes
        nested = false

        # Verificamos si la bounding box actual está dentro de otra
        for box2 in bounding_boxes
            if box1 != box2 && is_inside(box1, box2)
                nested = true
                break
            end
        end

        # Si la bounding box no está dentro de ninguna otra, la agregamos a la lista
        if !nested
            push!(non_nested_boxes, box1)
        end
    end

    return non_nested_boxes
end

# Función auxiliar para verificar si una bounding box está completamente contenida dentro de otra
function is_inside(box1, box2)
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    return x1 >= x3 && y1 >= y3 && x2 <= x4 && y2 <= y4
end



function join_near_bounding_boxes(bounding_boxes_list::Vector{T}, img, join_rate, color) where T<:Tuple
    # Creamos una lista para almacenar las bounding boxes unidas
    joined_boxes = Vector{T}()

    function increase_box(box, join_rate)
        
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1

        reduce_x = width * join_rate
        reduce_y = height * join_rate

        new_x1 = max(1, ceil(x1 - reduce_x))
        new_y1 = max(1, ceil(y1 - reduce_y))
        new_x2 = min(size(img, 2), ceil(x2 + reduce_x))
        new_y2 = min(size(img, 1), ceil(y2 + reduce_y))

        # Convertir a enteros
        new_x1 = Int(new_x1)
        new_y1 = Int(new_y1)
        new_x2 = Int(new_x2)
        new_y2 = Int(new_y2)

        return (new_x1, new_y1, new_x2, new_y2)
    end    

    # Aumentamos ligeramente cada bounding box
    for box in bounding_boxes_list
        increased_box = increase_box(box, join_rate)
        push!(joined_boxes, increased_box)
    end

    # Función auxiliar para verificar si dos bounding boxes están superpuestas
    function is_overlapping(box1, box2)
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        return (x1 <= x4 && x3 <= x2 && y1 <= y4 && y3 <= y2)
    end

    # Iteramos sobre todas las bounding boxes en la lista
    for box in joined_boxes
        # Flag para verificar si la bounding box actual se ha fusionado con otra
        joined = false

        # Iteramos sobre las bounding boxes unidas hasta ahora
        for (index, joined_box) in enumerate(joined_boxes)
            # Si la bounding box actual está superpuesta con alguna bounding box unida, fusionamos
            if is_overlapping(box, joined_box)
                # Fusionamos las bounding boxes
                new_box = (min(box[1], joined_box[1]), min(box[2], joined_box[2]), max(box[3], joined_box[3]), max(box[4], joined_box[4]))
                # Actualizamos la lista de bounding boxes unidas
                joined_boxes[index] = new_box
                # Marcamos la flag como verdadera para indicar que esta bounding box se ha fusionado
                joined = true
                break
            end
        end

        # Si la bounding box actual no se ha fusionado, la agregamos a la lista de bounding boxes unidas
        if !joined
            push!(joined_boxes, box)
        end
    end

    joined_boxes = remove_nested_boxes(joined_boxes)
    
    return joined_boxes
end


function print_bounding_boxes(boxes::Vector{T}) where T<:Tuple
    for (i, bbox) in enumerate(boxes)
        println("Bounding Box $i: ", bbox)
    end
end


