include("contornos.jl")

filename = "datos.txt"
file = open(filename, "w")

path = "Frames/frames1/"
images = 4

for i in 0:images
    newpath = path * "frame_$(i).jpg"
    println("\nNOW PROCESSING IMAGE $(i) ", newpath, "\n")
    local data = processImage(newpath)

    for element in data
        # Escribir datos
        write(file, "$(element[3]) $(element[4])\n")
    end
end

close(file)