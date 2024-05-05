include("../librerias.jl")
include("../crossval_funcs.jl")

seed!(1); # va a dar siempre los mismo si la arquitectura es la misma. Sirve para poder repetir experimentos y dean siempre los mismo.

numFolds = 10;

learningRate = 0.01;
numMaxEpochs = 1000;
validationRatio = 0.2;
maxEpochsVal = 6;
numRepetitionsANNTraining = 50;

kernel = "rbf";
kernelDegree = 3;
kernelGamma = 2;
C=1;
maxDepth = 21;

numNeighbors = 19;

dataset = readdlm("aprox1/datos_aprox1.data",',');

inputs = convert(Array{Float32,2}, dataset[:,1:end-1]);
targets = dataset[:,end];


crossValidationIndices = crossvalidation(targets, numFolds);
#normalizeZeroMean!(inputs);

topology = [16]
topology_string = join(topology,"+")

println("Comenzando entrenamiento con topología [$topology_string]")

dirname = "aprox_1"
output_name = dirname*"/$topology_string-output.txt"

if !isdir(dirname)
    mkdir(dirname);
end

modelHyperparameters = Dict();
modelHyperparameters["topology"] = topology;
modelHyperparameters["learningRate"] = learningRate;
modelHyperparameters["validationRatio"] = validationRatio;
modelHyperparameters["numExecutions"] = numRepetitionsANNTraining;
modelHyperparameters["maxEpochs"] = numMaxEpochs;
modelHyperparameters["maxEpochsVal"] = maxEpochsVal;


open(output_name,"w+") do out
    redirect_stdout(out) do
        println("RNA ----------------\n")
        @time "Elapsed time" modelCrossValidation(:ANN, modelHyperparameters, inputs, targets, crossValidationIndices);
    end
end


modelHyperparameters = Dict();
modelHyperparameters["kernel"] = kernel;
modelHyperparameters["kernelDegree"] = kernelDegree;
modelHyperparameters["kernelGamma"] = kernelGamma;
modelHyperparameters["C"] = C;
#println("\nSVM ----------------\n")
#modelCrossValidation(:SVM, modelHyperparameters, inputs, targets, crossValidationIndices);
#println("\nDT ----------------\n")
#modelCrossValidation(:DecisionTree, Dict("maxDepth" => maxDepth), inputs, targets, crossValidationIndices);
#println("\nkNN ----------------\n")
#modelCrossValidation(:kNN, Dict("numNeighbors" => numNeighbors), inputs, targets, crossValidationIndices);