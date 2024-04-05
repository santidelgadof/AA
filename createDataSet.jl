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
    
    for i in 0:dataSetSize
        imageFile = "frame_$(i).jpg"
        newpath = sourcePath * imageFile
        println("NOW PROCESSING IMAGE $(i)\n")
        local data = processImage(newpath, false)
        
    
        for element in data
            formatted3 = format(element[3], maxDecimals)
            formatted4 = format(element[4], maxDecimals)
            formatted5 = format(element[5], maxDecimals)
            
            if !idTags
                write(file, "$(formatted3), $(formatted4), $(formatted5), 0\n")
            else
                write(file, "$(formatted3), $(formatted4), $(formatted5), 0  \t\t| IMAGE: $i BOX: $(element[2])\n")
            end
        end

    end
    
    close(file)
end

createDataSet("Frames/frames2/", 69, "dataSetgen2.txt", true)