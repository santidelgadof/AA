include("contornos.jl")

function createDataSet(sourcePath, dataSetSize, filename)
    file = open(filename, "w")
    
    
    for i in 0:dataSetSize
        newpath = sourcePath * "frame_$(i).jpg"
        println("\nNOW PROCESSING IMAGE $(i)\n")
        local data = processImage(newpath)
    
        for element in data
            # Escribir datos
            write(file, "$(element[3]) $(element[4]), 0\n")
        end
    end
    
    close(file)
end