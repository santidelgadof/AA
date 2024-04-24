include("imageProcessing/contornos.jl")


function format(number, maxDecimals)
    str_num = string(round(number, digits=3))  # Redondear el número a tres decimales
    decimal_index = findfirst(x -> x == '.', str_num)

    if decimal_index === nothing
        decimals = 0
    else
        decimals = length(str_num) - decimal_index  # Número de decimales
    end

    zeros_to_add = max(0, maxDecimals - decimals)  # Calcular ceros a agregar, asegurándose de no ser negativo
    return "$(rpad(str_num, length(str_num) + zeros_to_add, '0'))"  # Añadir ceros a la derecha si es necesario
end


function createDataSet(sourcePath, dataSetSize, filename, idTags)

    maxDecimals = 3
    file = open(filename, "w")
    n=0
    
    for i in 0:dataSetSize
        imageFile = "frame_$(i).jpg"
        newpath = sourcePath * imageFile
        println("NOW PROCESSING IMAGE $(i)\n")
        local data = processImage(newpath, true)
        
        for element in data
            bounding_boxes = getBbox(newpath, "Bbox/", n)  # Aquí se llama a getBbox
            n+=1
            println(n)
            sizeRelations = format(element[3], maxDecimals)
            densities = format(element[4], maxDecimals)
            density_around_center = format(element[5], maxDecimals)
            density_in_each_side = format(element[6], maxDecimals)
            center_x, center_y = element[7]  # Desempaquetar las coordenadas del centro normalizado 

            rounded_center_x = round(center_x, digits=3)
            rounded_center_y = round(center_y, digits=3)
            # Obtener y guardar las bounding boxes
            

            if !idTags
                write(file, "$(sizeRelations), $(densities), $(density_around_center), 0\n")
            else
                write(file, "$(sizeRelations), $(densities), $(density_around_center), $(rounded_center_x), $(rounded_center_y), $(density_in_each_side), 0  \t\t| IMAGE: $i BOX: $(element[2])\n" )
            end
        end

    end
    
    close(file)
end

createDataSet("Frames/frames2/", 106, "dataset/datos5(datos).txt", true)