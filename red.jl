include("librerias.jl")
include("redfuncs.jl")





dataset = readdlm("datos.data",',');

inputs = dataset[:,1:2];
inputs = Float32.(inputs);
inputs = hcat(inputs);
targets = dataset[:,3];
targets = Bool.(targets);
targets =hcat(targets);


#ann = Chain(
# Dense(2, 4, σ),
# Dense(4, 1, σ) );
#
#loss(ann,x, y) = Flux.Losses.binarycrossentropy(ann(x), y)


#opt_state = Flux.setup(Adam(learningRate), ann)
# apotrofe traspone matriz
#Flux.train!(loss, ann, [(inputs', targets')], opt_state)



numInputs = 2
topology = [4]
transferFuncs = fill(σ, length(topology))
epoches = 100
minLoss = 0.0
learningRate = Float32.(0.03)

print("\n\nSTART\n\nINPUTS\n")
print(inputs)
print("\n\nOUTPUTS\n")
print(targets)
print("\n\nENTRENAMIENTO\n")

(ann, losses) = trainANN(topology, (inputs, targets), transferFuncs, epoches, minLoss, learningRate)
"END"



