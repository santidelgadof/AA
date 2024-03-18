using Flux
using Statistics
using FileIO;
using DelimitedFiles;

# Definir la arquitectura de la red neuronal
learningRate = 0.03


dataset = readdlm("datos.data",',');
# Ruta del archivo de datos
#filename = "datos.txt"

# Leer datos y etiquetas del archivo
inputs = dataset[:,1:2];
inputs = Float32.(inputs);
inputs = hcat(inputs);
targets = dataset[:,3];
targets = Float32.(targets);
targets =hcat(targets);
#targets = reshape(targets.==classes[1], :, 1);



ann = Chain(
 Dense(33, 4, σ),
 Dense(4, 1, σ) );



# Función de pérdida (loss function)
loss(ann,x, y) = Flux.Losses.binarycrossentropy(ann.(x), y)
 

# Optimizador
#optimizer = Flux.Optimise.ADAM()

opt_state = Flux.setup(Adam(learningRate), ann) # Inicializar el optimizador

Flux.train!(loss, ann, [(inputs, targets)], opt_state)
"""
# Función para entrenar la red neuronal
function train!(ann, X, y, optimizer; epochs=10, batch_size=32)
    dataset = Flux.Data.DataLoader((X, y), batchsize=batch_size, shuffle=true)
    for epoch in 1:epochs
        for (X_batch, y_batch) in dataset
            Flux.train!(loss, Flux.params(ann), [(X_batch, y_batch)], optimizer)
        end
        println("Epoch epoch, Loss: (loss(X_batch, y_batch))")
    end
end

# Función para evaluar la precisión del modelo
function accuracy(ann, X, y)
    ŷ = ann(X)
    y_pred = ifelse.(ŷ .>= 0.5, 1, 0)
    acc = mean(y_pred .== y)
    return acc
end

# Leer datos de un archivo
function read_data(filename)
    data = []
    labels = []
    open(filename) do file
        for line in eachline(file)
            parts = split(line, ',')
            push!(data, parse(Float64, parts[1]))  # Solo tomar densidad 
            push!(labels, parse(Int, parts[2]))       # Convertir la última parte a Int
        end
    end
    return data, labels
end


# EJEMPLO ARCHIVO TXT

# 0.2654682274247492,0
# 0.10707627118644068,0
# 0.046413502109704644,0
# 0.5833333333333334,1
# 0.6241379310344828,0
# 0.5238095238095237,1
# 0.4625,0

# (En esta estructura, cada fila representa un bounding box de una imagen)
# El último valor es la etiqueta que indica si hay o no un enemigo en el bounding box (0 o 1).


#X_train, y_train = read_data(filename)

# Entrenamiento de la red neuronal
#train!(ann, X_train, y_train, optimizer, epochs=10)



# Evaluación del modelo (si tienes datos de prueba)
# Proporciona aquí tus datos de prueba (si los tienes) para evaluar la precisión del modelo
# X_test, y_test = read_data("datos_de_prueba.txt")
# acc = accuracy(ann, X_test, y_test)
# println("Accuracy en conjunto de prueba: acc")