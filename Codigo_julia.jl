using Images, ImageFiltering, FileIO, ImageView, ImageMorphology, ColorTypes, ImageDraw, Colors

#Load image
img = load("Frames/edition1/frame23_194.png")
img_hsv = HSV.(img)

# Define the yellow color range in HSV, ensuring the type matches img_hsv pixels
yellow_low = HSV{Float32}(40, 100/255, 155/255)
yellow_high = HSV{Float32}(71, 255/255, 255/255)

#Generate mask
mask = map(img_hsv) do pixel
    h, s, v = pixel.h, pixel.s, pixel.v
    h_low, s_low, v_low = yellow_low.h, yellow_low.s, yellow_low.v
    h_high, s_high, v_high = yellow_high.h, yellow_high.s, yellow_high.v

    h_in_range = h_low <= h <= h_high
    s_in_range = s_low <= s <= s_high
    v_in_range = v_low <= v <= v_high

    return h_in_range && s_in_range && v_in_range ? 1 : 0
end

# Convert mask to grayscale image for morphological operations
mask_gray = Gray.(mask)

# Convolution matrices
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

#img_erode = @. Gray(mask_gray < 0.1); # keeps white objects white
img_erosion1 = closing(mask_gray,se2)

#Bounding boxes function
function bounding_boxes(mask::Array{T, 2}) where {T}
    # Find objects in the mask
    objects = label_components(mask)
    
    # Initialize a vector to store the bounding boxes of each object
    bounding_boxes_list = Tuple[]
    
    # Iterate over each object and calculate its bounding box
    for obj in 1:maximum(objects)
        # Create a mask for the current object
        obj_mask = objects .== obj
        
        # Finding the indices of non-black pixels in the object mask
        non_black_pixels = findall(!isequal(zero(T)), obj_mask)
        
        # Calculate minimum and maximum coordinates of non-black pixels
        min_x = minimum(i[2] for i in non_black_pixels)
        max_x = maximum(i[2] for i in non_black_pixels)
        min_y = minimum(i[1] for i in non_black_pixels)
        max_y = maximum(i[1] for i in non_black_pixels)
        
        # Add the bounding box to the list
        push!(bounding_boxes_list, (min_x, min_y, max_x, max_y))
    end
    
    # Return list with bounding boxes
    return bounding_boxes_list
end

#Bounding boxes image
bbox = bounding_boxes(img_erosion1)

# Merge the Bounding boxes
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

#Prints bounding boxes
function print_merged_bounding_boxes(merged_boxes::Vector{T}) where T<:Tuple
    for (i, bbox) in enumerate(merged_boxes)
        println("Bounding Box $i: ", bbox)
    end
end

# Definir umbral de proximidad para la fusión de bounding boxes (ajusta este valor según tus necesidades)
threshold = 100  

# Fusionar las bounding boxes cercanas
merged_boxes = merge_close_bounding_boxes(bbox, threshold)


# Imprimir las bounding boxes fusionadas
#print_merged_bounding_boxes(merged_boxes)

# Draws bounding boxes
function draw_bounding_boxes(bounding_boxes_list::Vector{T}, img, color) where T<:Tuple
    # Make a deep copy of the image to avoid modifying the original
    img2 = deepcopy(img)
    
    # Iterate over all bounding boxes in the list
    for i in 1:length(bounding_boxes_list)
        # Extract the coordinates of the current bounding box
        x1, y1, x2, y2 = bounding_boxes_list[i]
        
        # Draw the horizontal borders of the bounding box
        for x in x1:x2
            img2[y1, x] = color
            img2[y2, x] = color
        end
        
        # Draw the vertical borders of the bounding box
        for y in y1:y2
            img2[y, x1] = color
            img2[y, x2] = color
        end
    end
    
    # Return the image with the bounding boxes drawn
    return img2
end

# Reduce bounding boxes
function reduce_bounding_boxes(bounding_boxes_list::Vector{T}, threshold) where T<:Tuple
    # Create a copy of the list of bounding boxes to avoid modifying the original
    copy_list = copy(bounding_boxes_list)
    
    # Initialize a counter to keep track of deletions
    counter = 0
    
    # Iterate over all bounding boxes in the list
    for i in 1:length(bounding_boxes_list)
        # Extract the coordinates of the current bounding box
        x1, y1, x2, y2 = bounding_boxes_list[i]
        
        # Calculate the height and width of the bounding box
        height = y2 - y1
        width = x2 - x1
        
        # Calculate the ratio between width and height of the bounding box
        x_proportion = width / height
        
        # Check if the bounding box meets any of the conditions to be deleted
        if ((height < threshold && width < threshold) || x_proportion > 5 || x_proportion < 0.2)
            # If the bounding box does not meet the conditions, remove it from the copied list
            deleteat!(copy_list, i - counter)
            
            # Increment the deletion counter
            counter += 1
        end
    end
    
    # Return the list of bounding boxes after applying the deletions
    return copy_list
end

reduced_boxes= reduce_bounding_boxes(merged_boxes,15)
img_bound = draw_bounding_boxes(reduced_boxes, img_erosion1, 1)

#Removes nested boxes
function remove_nested_boxes(bounding_boxes::Vector{T}) where T<:Tuple
    # Create a list to store the bounding boxes that are not inside others
    non_nested_boxes = Vector{T}()

    # Iterate over all bounding boxes
    for box1 in bounding_boxes
        nested = false

        # Check if the current bounding box is inside another
        for box2 in bounding_boxes
            if box1 != box2 && is_inside(box1, box2)
                nested = true
                break
            end
        end

        # If the bounding box is not inside any other, add it to the list
        if !nested
            push!(non_nested_boxes, box1)
        end
    end

    return non_nested_boxes
end

# Helper function to check if a bounding box is completely contained within another
function is_inside(box1, box2)
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    return x1 >= x3 && y1 >= y3 && x2 <= x4 && y2 <= y4
end

# Convert near bounding boxes in only one 
function join_near_bounding_boxes(bounding_boxes_list::Vector{T}, img, threshold, color) where T<:Tuple
    # Create a list to store the joined bounding boxes
    joined_boxes = Vector{T}()

    function increase_box(box, threshold)
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        new_x1 = max(1, x1 - threshold)
        new_y1 = max(1, y1 - threshold)
        new_x2 = min(size(img, 2), x2 + threshold)
        new_y2 = min(size(img, 1), y2 + threshold)
        return (new_x1, new_y1, new_x2, new_y2)
    end    

    # Slightly increase each bounding box
    for box in bounding_boxes_list
        increased_box = increase_box(box, threshold)
        push!(joined_boxes, increased_box)
    end

    # Auxiliary function to check if two bounding boxes overlap
    function is_overlapping(box1, box2)
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        return (x1 <= x4 && x3 <= x2 && y1 <= y4 && y3 <= y2)
    end

    # Iterate over all bounding boxes in the list
    for box in joined_boxes
        # Flag to check if the current bounding box has been merged with another
        joined = false

        # Iterate over the joined bounding boxes so far
        for (index, joined_box) in enumerate(joined_boxes)
            # If the current bounding box overlaps with any joined bounding box, merge
            if is_overlapping(box, joined_box)
                # Merge the bounding boxes
                new_box = (min(box[1], joined_box[1]), min(box[2], joined_box[2]), max(box[3], joined_box[3]), max(box[4], joined_box[4]))
                # Update the list of joined bounding boxes
                joined_boxes[index] = new_box
                # Set the flag to true to indicate that this bounding box has been merged
                joined = true
                break
            end
        end

        # If the current bounding box has not been merged, add it to the list of joined bounding boxes
        if !joined
            push!(joined_boxes, box)
        end
    end

    joined_boxes = remove_nested_boxes(joined_boxes)

    # Draw the joined bounding boxes on the image using the existing function
    image_with_joined_boxes = draw_bounding_boxes(joined_boxes, img, color)

    return image_with_joined_boxes
end

img_unida = join_near_bounding_boxes(reduced_boxes, img_erosion1, 10, 1)

# Calculates bounding boxes density
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

total_image_area = size(img_erosion1, 1) * size(img_erosion1, 2)  # Calcular el área total de la imagen
densities = bounding_box_density(reduced_boxes, total_image_area)

