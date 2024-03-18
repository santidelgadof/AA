using Flux
using Statistics
using FileIO;
using DelimitedFiles;


learningRate = 0.03


dataset = readdlm("datos.data",',');
# Ruta del archivo de datos

# Leer datos y etiquetas del archivo
inputs = dataset[:,1:2];
inputs = Float32.(inputs);
inputs = hcat(inputs);
targets = dataset[:,3];
targets = Float32.(targets);
targets =hcat(targets);
#targets = reshape(targets.==classes[1], :, 1);

targets
inputs

ann = Chain(
 Dense(2, 4, σ),
 Dense(4, 1, σ) );


# Función de pérdida (loss function)
loss(ann,x, y) = Flux.Losses.binarycrossentropy(ann(x), y)
 

opt_state = Flux.setup(Adam(learningRate), ann) # Inicializar el optimizador
# apotrofe traspone matriz
Flux.train!(loss, ann, [(inputs', targets')], opt_state)
