include("contornos.jl")

function createDataSet(sourcePath, dataSetSize, filename, idTags)
    file = open(filename, "w")
    
    
    for i in 0:dataSetSize
        imageFile = "frame_$(i).jpg"
        newpath = sourcePath * imageFile
        println("NOW PROCESSING IMAGE $(i)\n")
        local data = processImage(newpath)
    
        for element in data
            # Escribir datos
            if !idTags
                write(file, "$(element[3]) $(element[4]), 0\n")
            else
                write(file, "$(element[3]) $(element[4]), 0  \t\t| IMAGE: $i BOX: $(element[2])\n")
            end
        end
    end
    
    close(file)
end