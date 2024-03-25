include("contornos.jl")

function format(number, maxDecimals)
    str_num = string(number)
    decimal_index = findfirst(x -> x == '.', str_num)

    if decimal_index === nothing
        decimals = 0  # No hay decimales
    else
        decimals = length(str_num) - decimal_index  # Longitud de la parte decimal después del punto
    end

    zeros_to_add = maxDecimals - decimals  # Calcula el número de ceros a agregar
    return "$number"*("0"^zeros_to_add)  # Crea una cadena de ceros
end


function createDataSet(sourcePath, dataSetSize, filename, idTags)

    maxDecimals = 20
    file = open(filename, "w")
    
    for i in 0:dataSetSize
        imageFile = "frame_$(i).jpg"
        newpath = sourcePath * imageFile
        println("NOW PROCESSING IMAGE $(i)\n")
        local data = processImage(newpath)
        
    
        for element in data
            formatted3 = format(element[3], maxDecimals)
            formatted4 = format(element[4], maxDecimals)
            
            if !idTags
                write(file, "$(formatted3) $(formatted4), 0\n")
            else
                write(file, "$(formatted3) $(formatted4), 0  \t\t| IMAGE: $i BOX: $(element[2])\n")
            end
        end

    end
    
    close(file)
end