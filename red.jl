include("librerias.jl")
include("redfuncs.jl")
include("createDataSet.jl")

filename = "bounding_enemy.txt"
sourcePath = "frames/frames2/"
dataSetSize = 69
idTags = true

#createDataSet(sourcePath, dataSetSize, filename, idTags)

dataset = readdlm("dataset/datos.data",',');

inputs = dataset[:,1:2];
inputs = Float32.(inputs);
inputs = hcat(inputs);
targets = dataset[:,3];
targets = Bool.(targets);
targets =hcat(targets);

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



