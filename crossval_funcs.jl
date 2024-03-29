include("librerias.jl")
@sk_import svm: SVC
@sk_import tree: DecisionTreeClassifier
@sk_import neighbors: KNeighborsClassifier

function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    @assert(all([in(value, classes) for value in feature]));
    numClasses = length(classes);
    @assert(numClasses>1)
    if (numClasses==2)
        oneHot = reshape(feature.==classes[1], :, 1);
    else
        oneHot =  BitArray{2}(undef, length(feature), numClasses);
        for numClass = 1:numClasses
            oneHot[:,numClass] .= (feature.==classes[numClass]);
        end;
    end;
    return oneHot;
end;
oneHotEncoding(feature::AbstractArray{<:Any,1}) = oneHotEncoding(feature, unique(feature));
oneHotEncoding(feature::AbstractArray{Bool,1}) = reshape(feature, :, 1);

calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2}) = ( minimum(dataset, dims=1), maximum(dataset, dims=1) );
calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real,2}) = ( mean(dataset, dims=1), std(dataset, dims=1) );

function normalizeMinMax!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    minValues = normalizationParameters[1];
    maxValues = normalizationParameters[2];
    dataset .-= minValues;
    dataset ./= (maxValues .- minValues);
    dataset[:, vec(minValues.==maxValues)] .= 0;
    return dataset;
end;
normalizeMinMax!(dataset::AbstractArray{<:Real,2})                                                              = normalizeMinMax!(     dataset , calculateMinMaxNormalizationParameters(dataset));
normalizeMinMax( dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}}) = normalizeMinMax!(copy(dataset), normalizationParameters)
normalizeMinMax( dataset::AbstractArray{<:Real,2})                                                              = normalizeMinMax!(copy(dataset), calculateMinMaxNormalizationParameters(dataset));

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    avgValues = normalizationParameters[1];
    stdValues = normalizationParameters[2];
    dataset .-= avgValues;
    dataset ./= stdValues;
    dataset[:, vec(stdValues.==0)] .= 0;
    return dataset;
end;
normalizeZeroMean!(dataset::AbstractArray{<:Real,2})                                                              = normalizeZeroMean!(     dataset , calculateZeroMeanNormalizationParameters(dataset));
normalizeZeroMean( dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}}) = normalizeZeroMean!(copy(dataset), normalizationParameters)
normalizeZeroMean( dataset::AbstractArray{<:Real,2})                                                              = normalizeZeroMean!(copy(dataset), calculateZeroMeanNormalizationParameters(dataset));

function classifyOutputs(outputs::AbstractArray{<:Real,2}; threshold::Real=0.5)
    numOutputs = size(outputs, 2);
    @assert(numOutputs!=2)
    if numOutputs==1
        return outputs.>=threshold;
    else
        (_,indicesMaxEachInstance) = findmax(outputs, dims=2);
        outputs = falses(size(outputs));
        outputs[indicesMaxEachInstance] .= true;
        @assert(all(sum(outputs, dims=2).==1));
        return outputs;
    end;
end;

accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1}) = mean(outputs.==targets);
function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2})
    @assert(all(size(outputs).==size(targets)));
    if (size(targets,2)==1)
        return accuracy(outputs[:,1], targets[:,1]);
    else
        return mean(all(targets .== outputs, dims=2));
    end;
end;

accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5) = accuracy(outputs.>=threshold, targets);
function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5)
    @assert(all(size(outputs).==size(targets)));
    if (size(targets,2)==1)
        return accuracy(outputs[:,1], targets[:,1]);
    else
        return accuracy(classifyOutputs(outputs; threshold=threshold), targets);
    end;
end;

function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)))
    ann=Chain();
    numInputsLayer = numInputs;
    for numHiddenLayer in 1:length(topology)
        numNeurons = topology[numHiddenLayer];
        ann = Chain(ann..., Dense(numInputsLayer, numNeurons, transferFunctions[numHiddenLayer]));
        numInputsLayer = numNeurons;
    end;
    if (numOutputs == 1)
        ann = Chain(ann..., Dense(numInputsLayer, 1, σ));
    else
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity));
        ann = Chain(ann..., softmax);
    end;
    return ann;
end;

function trainClassANN(topology::AbstractArray{<:Int,1}, dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}; 
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)
    (inputs, targets) = dataset;
    @assert(size(inputs,1)==size(targets,1));
    ann = buildClassANN(size(inputs,2), topology, size(targets,2));
    loss(x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(ann(x),y) : Losses.crossentropy(ann(x),y);
    trainingLosses = Float32[];
    numEpoch = 0;
    trainingLoss = loss(inputs', targets');
    push!(trainingLosses, trainingLoss);
    println("Epoch ", numEpoch, ": loss: ", trainingLoss);

    while (numEpoch<maxEpochs) && (trainingLoss>minLoss)
        Flux.train!(loss, Flux.params(ann), [(inputs', targets')], ADAM(learningRate));

        numEpoch += 1;
        trainingLoss = loss(inputs', targets');
        push!(trainingLosses, trainingLoss);
        println("Epoch ", numEpoch, ": loss: ", trainingLoss);
    end;

    return (ann, trainingLosses);
end;

trainClassANN(topology::AbstractArray{<:Int,1}, (inputs, targets)::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01) = trainClassANN(topology, (inputs, reshape(targets, length(targets), 1)); maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate)

function holdOut(N::Int, P::Real)
    @assert ((P>=0.) & (P<=1.));
    indices = randperm(N);
    numTrainingInstances = Int(round(N*(1-P)));
    return (indices[1:numTrainingInstances], indices[numTrainingInstances+1:end]);
end

function holdOut(N::Int, Pval::Real, Ptest::Real)
    @assert ((Pval>=0.) & (Pval<=1.));
    @assert ((Ptest>=0.) & (Ptest<=1.));
    @assert ((Pval+Ptest)<=1.);
    (trainingValidationIndices, testIndices) = holdOut(N, Ptest);
    (trainingIndices, validationIndices) = holdOut(length(trainingValidationIndices), Pval*N/length(trainingValidationIndices))
    return (trainingValidationIndices[trainingIndices], trainingValidationIndices[validationIndices], testIndices);
end;

function trainClassANN(topology::AbstractArray{<:Int,1}, trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)),
    testDataset::      Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20, showText::Bool=false)

    (trainingInputs,   trainingTargets)   = trainingDataset;
    (validationInputs, validationTargets) = validationDataset;
    (testInputs,       testTargets)       = testDataset;

    @assert(size(trainingInputs,   1)==size(trainingTargets,   1));
    @assert(size(testInputs,       1)==size(testTargets,       1));
    @assert(size(validationInputs, 1)==size(validationTargets, 1));
    !isempty(validationInputs)  && @assert(size(trainingInputs, 2)==size(validationInputs, 2));
    !isempty(validationTargets) && @assert(size(trainingTargets,2)==size(validationTargets,2));
    !isempty(testInputs)  && @assert(size(trainingInputs, 2)==size(testInputs, 2));
    !isempty(testTargets) && @assert(size(trainingTargets,2)==size(testTargets,2));

    ann = buildClassANN(size(trainingInputs,2), topology, size(trainingTargets,2); transferFunctions=transferFunctions);
    loss(x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(ann(x),y) : Losses.crossentropy(ann(x),y);

    trainingLosses   = Float32[];
    validationLosses = Float32[];
    testLosses       = Float32[];

    numEpoch = 0;

    function calculateLossValues()
        trainingLoss = loss(trainingInputs', trainingTargets');
        showText && print("Epoch ", numEpoch, ": Training loss: ", trainingLoss);
        push!(trainingLosses, trainingLoss);
        if !isempty(validationInputs)
            validationLoss = loss(validationInputs', validationTargets');
            showText && print(" - validation loss: ", validationLoss);
            push!(validationLosses, validationLoss);
        else
            validationLoss = NaN;
        end;
        if !isempty(testInputs)
            testLoss       = loss(testInputs', testTargets');
            showText && print(" - test loss: ", testLoss);
            push!(testLosses, testLoss);
        else
            testLoss = NaN;
        end;
        showText && println("");
        return (trainingLoss, validationLoss, testLoss);
    end;

    (trainingLoss, validationLoss, _) = calculateLossValues();

    numEpochsValidation = 0; bestValidationLoss = validationLoss;
    bestANN = deepcopy(ann);

    while (numEpoch<maxEpochs) && (trainingLoss>minLoss) && (numEpochsValidation<maxEpochsVal)
        Flux.train!(loss, Flux.params(ann), [(trainingInputs', trainingTargets')], ADAM(learningRate));
        numEpoch += 1;
        (trainingLoss, validationLoss, _) = calculateLossValues();

        if (!isempty(validationInputs))
            if (validationLoss<bestValidationLoss)
                bestValidationLoss = validationLoss;
                numEpochsValidation = 0;
                bestANN = deepcopy(ann);
            else
                numEpochsValidation += 1;
            end;
        end;
    end;

    if isempty(validationInputs)
        bestANN = ann;
    end;

    return (bestANN, trainingLosses, validationLosses, testLosses);
end;


function trainClassANN(topology::AbstractArray{<:Int,1}, trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),1}(undef,0,0), falses(0)),
    testDataset::      Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),1}(undef,0,0), falses(0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20, showText::Bool=false)

    (trainingInputs,   trainingTargets)   = trainingDataset;
    (validationInputs, validationTargets) = validationDataset;
    (testInputs,       testTargets)       = testDataset;

    return trainClassANN(topology, (trainingInputs, reshape(trainingTargets, length(trainingTargets), 1)); validationDataset=(validationInputs, reshape(validationTargets, length(validationTargets), 1)), testDataset=(testInputs, reshape(testTargets, length(testTargets), 1)), transferFunctions=transferFunctions, maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate, maxEpochsVal=maxEpochsVal, showText=showText);
end;

function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    numInstances = length(targets);
    @assert(length(outputs)==numInstances);
    TN = sum(.!outputs .& .!targets); # Verdaderos negativos
    FN = sum(.!outputs .&   targets); # Falsos negativos
    TP = sum(  outputs .&   targets); # Verdaderos positivos
    FP = sum(  outputs .& .!targets); # Falsos negativos
    confMatrix = [TN FP; FN TP];
    acc         = (TN+TP)/(TN+FN+TP+FP);
    errorRate   = 1. - acc;
    if (TN==numInstances) || (TP==numInstances)
        recall = 1.;
        precision = 1.;
        specificity = 1.;
        NPV = 1.;
    else
        recall      = (TP==TP==0.) ? 0. : TP/(TP+FN); # Sensibilidad
        specificity = (TN==FP==0.) ? 0. : TN/(TN+FP); # Especificidad
        precision   = (TP==FP==0.) ? 0. : TP/(TP+FP); # Valor predictivo positivo
        NPV         = (TN==FN==0.) ? 0. : TN/(TN+FN); # Valor predictivo negativo
    end;
    F1          = (recall==precision==0.) ? 0. : 2*(recall*precision)/(recall+precision);
    return (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix)
end;

confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5) = confusionMatrix(outputs.>=threshold, targets);


function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    @assert(size(outputs)==size(targets));
    numClasses = size(targets,2);
    @assert(numClasses!=2);
    if (numClasses==1)
        return confusionMatrix(outputs[:,1], targets[:,1]);
    end;

    @assert(all(sum(outputs, dims=2).==1));
    recall      = zeros(numClasses);
    specificity = zeros(numClasses);
    precision   = zeros(numClasses);
    NPV         = zeros(numClasses);
    F1          = zeros(numClasses);
    numInstancesFromEachClass = vec(sum(targets, dims=1));
    for numClass in findall(numInstancesFromEachClass.>0)
        (_, _, recall[numClass], specificity[numClass], precision[numClass], NPV[numClass], F1[numClass], _) = confusionMatrix(outputs[:,numClass], targets[:,numClass]);
    end;

    confMatrix = Array{Int64,2}(undef, numClasses, numClasses);
    for numClassTarget in 1:numClasses, numClassOutput in 1:numClasses
        confMatrix[numClassTarget, numClassOutput] = sum(targets[:,numClassTarget] .& outputs[:,numClassOutput]);
    end;

    if weighted
        weights = numInstancesFromEachClass./sum(numInstancesFromEachClass);
        recall      = sum(weights.*recall);
        specificity = sum(weights.*specificity);
        precision   = sum(weights.*precision);
        NPV         = sum(weights.*NPV);
        F1          = sum(weights.*F1);
    else
        numClassesWithInstances = sum(numInstancesFromEachClass.>0);
        recall      = sum(recall)/numClassesWithInstances;
        specificity = sum(specificity)/numClassesWithInstances;
        precision   = sum(precision)/numClassesWithInstances;
        NPV         = sum(NPV)/numClassesWithInstances;
        F1          = sum(F1)/numClassesWithInstances;
    end;
    acc = accuracy(outputs, targets);
    errorRate = 1 - acc;

    return (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix);
end;

confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true) = confusionMatrix(classifyOutputs(outputs), targets; weighted=weighted)



function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    @assert(all([in(output, unique(targets)) for output in outputs]));
    classes = unique([targets; outputs]);
    return confusionMatrix(oneHotEncoding(outputs, classes), oneHotEncoding(targets, classes); weighted=weighted);
end;

function printConfusionMatrix(confMatrix::Matrix{Int64})
    numClasses = size(confMatrix,1);
    writeHorizontalLine() = (for i in 1:numClasses+1 print("--------") end; println(""); );
    writeHorizontalLine();
    print("\t| ");
    if (numClasses==2)
        println(" - \t + \t|");
    else
        print.("Cl. ", 1:numClasses, "\t| ");
    end;
    println("");
    writeHorizontalLine();
    for numClassTarget in 1:numClasses
        # print.(confMatrix[numClassTarget,:], "\t");
        if (numClasses==2)
            print(numClassTarget == 1 ? " - \t| " : " + \t| ");
        else
            print("Cl. ", numClassTarget, "\t| ");
        end;
        print.(confMatrix[numClassTarget,:], "\t| ");
        println("");
        writeHorizontalLine();
    end;
end;
function printConfusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix) = confusionMatrix(outputs, targets; weighted=weighted);
    numClasses = size(confMatrix,1);
    writeHorizontalLine() = (for i in 1:numClasses+1 print("--------") end; println(""); );
    writeHorizontalLine();
    print("\t| ");
    if (numClasses==2)
        println(" - \t + \t|");
    else
        print.("Cl. ", 1:numClasses, "\t| ");
    end;
    println("");
    writeHorizontalLine();
    for numClassTarget in 1:numClasses
        # print.(confMatrix[numClassTarget,:], "\t");
        if (numClasses==2)
            print(numClassTarget == 1 ? " - \t| " : " + \t| ");
        else
            print("Cl. ", numClassTarget, "\t| ");
        end;
        print.(confMatrix[numClassTarget,:], "\t| ");
        println("");
        writeHorizontalLine();
    end;
    println("Accuracy: ", acc);
    println("Error rate: ", errorRate);
    println("Recall: ", recall);
    println("Specificity: ", specificity);
    println("Precision: ", precision);
    println("Negative predictive value: ", NPV);
    println("F1-score: ", F1);
    return (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix);
end;
printConfusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true) =  printConfusionMatrix(classifyOutputs(outputs), targets; weighted=weighted)
printConfusionMatrix(outputs::Matrix{Int64}, targets::Matrix{Int64}; weighted::Bool=true) =  printConfusionMatrix(classifyOutputs(outputs), targets; weighted=weighted)
printConfusionMatrix(outputs::AbstractArray{Bool,1},   targets::AbstractArray{Bool,1})                      = printConfusionMatrix(reshape(outputs, :, 1), targets);
printConfusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5) = printConfusionMatrix(outputs.>=threshold,    targets);

function crossvalidation(N::Int64, k::Int64)
    indices = repeat(1:k, Int64(ceil(N/k)));
    indices = indices[1:N];
    shuffle!(indices);
    return indices;
end;

function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    numClasses = size(targets,2);
    indices = Array{Int64,1}(undef, size(targets,1));
    for numClass in 1:numClasses
        indices[targets[:,numClass]] = crossvalidation(sum(targets[:,numClass]), k);
    end;
    return indices;
end;

function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
    classes = unique(targets);
    indices = Array{Int64,1}(undef, length(targets));
    for class in classes
        indicesThisClass = (targets .== class);
        indices[indicesThisClass] = crossvalidation(sum(indicesThisClass), k);
    end;
    return indices;
end;

function trainClassANN(topology::AbstractArray{<:Int,1}, trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}},
    kFoldIndices::     Array{Int64,1};
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
    numRepetitionsANNTraining::Int=1, validationRatio::Real=0.0,
    maxEpochsVal::Int=20)

    numFolds = maximum(kFoldIndices);

    testAccuracies = Array{Float64,1}(undef, numFolds);
    testF1         = Array{Float64,1}(undef, numFolds);

    for numFold in 1:numFolds
        trainingInputs    = inputs[kFoldIndices.!=numFold,:];
        testInputs        = inputs[kFoldIndices.==numFold,:];
        trainingTargets   = targets[kFoldIndices.!=numFold,:];
        testTargets       = targets[kFoldIndices.==numFold,:];

        testAccuraciesEachRepetition = Array{Float64,1}(undef, numRepetitionsANNTraining);
        testF1EachRepetition         = Array{Float64,1}(undef, numRepetitionsANNTraining);

        for numTraining in 1:numRepetitionsANNTraining
            if validationRatio > 0
                ann, = trainClassANN(topology, (trainingInputs[trainingIndices,:],   trainingTargets[trainingIndices,:]),
                    validationDataset = (trainingInputs[validationIndices,:], trainingTargets[validationIndices,:]),
                    testDataset =       (testInputs,                          testTargets);
                    maxEpochs=numMaxEpochs, learningRate=learningRate, maxEpochsVal=maxEpochsVal);
            else
                ann, = trainClassANN(topology, (trainingInputs, trainingTargets),
                    testDataset = (testInputs,     testTargets);
                    maxEpochs=numMaxEpochs, learningRate=learningRate);
            end;
            (acc, _, _, _, _, _, F1, _) = confusionMatrix(ann(testInputs')', testTargets);
            testAccuraciesEachRepetition[numTraining] = acc;
            testF1EachRepetition[numTraining]         = F1;
        end;
        testAccuracies[numFold] = mean(testAccuraciesEachRepetition);
        testF1[numFold]         = mean(testF1EachRepetition);
        println("Results in test in fold ", numFold, "/", numFolds, ": accuracy: ", 100*testAccuracies[numFold], " %, F1: ", 100*testF1[numFold], " %");
    end;
    println("Average test accuracy on a ", numFolds, "-fold crossvalidation: ", 100*mean(testAccuracies), ", with a standard deviation of ", 100*std(testAccuracies));
    println("Average test F1 on a ", numFolds, "-fold crossvalidation: ", 100*mean(testF1), ", with a standard deviation of ", 100*std(testF1));
end;


function trainClassANN(topology::AbstractArray{<:Int,1}, trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}},
    kFoldIndices::     Array{Int64,1};
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
    numRepetitionsANNTraining::Int=1, validationRatio::Real=0.0,
    maxEpochsVal::Int=20)

    (trainingInputs,   trainingTargets)   = trainingDataset;

    return trainClassANN(topology, (trainingInputs, reshape(trainingTargets, length(trainingTargets), 1)), kFoldIndices; transferFunctions=transferFunctions, maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate, maxEpochsVal=maxEpochsVal);

end;

function modelCrossValidation(modelType::Symbol, modelHyperparameters::Dict, inputs::AbstractArray{<:Real,2}, targets::AbstractArray{<:Any,1}, crossValidationIndices::Array{Int64,1})
    @assert(size(inputs,1)==length(targets));
    classes = unique(targets);

    if modelType==:ANN
        targets = oneHotEncoding(targets, classes);
    end;

    testAccuracies = Array{Float64,1}(undef, numFolds);
    testF1         = Array{Float64,1}(undef, numFolds);

    local ann_matrix;

    for numFold in 1:numFolds
        if (modelType==:SVM) || (modelType==:DecisionTree) || (modelType==:kNN)
            trainingInputs    = inputs[crossValidationIndices.!=numFold,:];
            testInputs        = inputs[crossValidationIndices.==numFold,:];
            trainingTargets   = targets[crossValidationIndices.!=numFold];
            testTargets       = targets[crossValidationIndices.==numFold];

            if modelType==:SVM
                model = SVC(kernel=modelHyperparameters["kernel"], degree=modelHyperparameters["kernelDegree"], gamma=modelHyperparameters["kernelGamma"], C=modelHyperparameters["C"]);
            elseif modelType==:DecisionTree
                model = DecisionTreeClassifier(max_depth=modelHyperparameters["maxDepth"], random_state=1);
            elseif modelType==:kNN
                model = KNeighborsClassifier(modelHyperparameters["numNeighbors"]);
            end;

            model = ScikitLearn.fit!(model, trainingInputs, trainingTargets);
            testOutputs = ScikitLearn.predict(model, testInputs);
            (acc, _, _, _, _, _, F1, cmatrix) = confusionMatrix(testOutputs, testTargets);
            printConfusionMatrix(cmatrix);
        else
            trainingInputs    = inputs[crossValidationIndices.!=numFold,:];
            testInputs        = inputs[crossValidationIndices.==numFold,:];
            trainingTargets   = targets[crossValidationIndices.!=numFold,:];
            testTargets       = targets[crossValidationIndices.==numFold,:];

            testAccuraciesEachRepetition = Array{Float64,1}(undef, modelHyperparameters["numExecutions"]);
            testF1EachRepetition         = Array{Float64,1}(undef, modelHyperparameters["numExecutions"]);

            for numTraining in 1:modelHyperparameters["numExecutions"]
                if modelHyperparameters["validationRatio"] > 0
                    (trainingIndices, validationIndices) = holdOut(size(trainingInputs,1), modelHyperparameters["validationRatio"]*size(trainingInputs,1)/size(inputs,1));
                    ann, = trainClassANN(modelHyperparameters["topology"], (trainingInputs[trainingIndices,:],   trainingTargets[trainingIndices,:]),
                        validationDataset = (trainingInputs[validationIndices,:], trainingTargets[validationIndices,:]),
                        testDataset =       (testInputs,                          testTargets);
                        maxEpochs=modelHyperparameters["maxEpochs"], learningRate=modelHyperparameters["learningRate"], maxEpochsVal=modelHyperparameters["maxEpochsVal"]);
                else
                    ann, = trainClassANN(modelHyperparameters["topology"], (trainingInputs, trainingTargets),
                        testDataset = (testInputs,     testTargets);
                        maxEpochs=modelHyperparameters["maxEpochs"], learningRate=modelHyperparameters["learningRate"]);
                end;

                (testAccuraciesEachRepetition[numTraining], _, _, _, _, _, testF1EachRepetition[numTraining], cmatrix) = confusionMatrix(collect(ann(testInputs')'), testTargets);
                ann_matrix = cmatrix;
            end;

            acc = mean(testAccuraciesEachRepetition);
            F1  = mean(testF1EachRepetition);
            printConfusionMatrix(ann_matrix);
        end;

        testAccuracies[numFold] = acc;
        testF1[numFold]         = F1;

        println("Results in test in fold ", numFold, "/", numFolds, ": accuracy: ", 100*testAccuracies[numFold], " %, F1: ", 100*testF1[numFold], " %");
    end;

    println(modelType, ": Average test accuracy on a ", numFolds, "-fold crossvalidation: ", 100*mean(testAccuracies), ", with a standard deviation of ", 100*std(testAccuracies));
    println(modelType, ": Average test F1 on a ", numFolds, "-fold crossvalidation: ", 100*mean(testF1), ", with a standard deviation of ", 100*std(testF1));

    return (mean(testAccuracies), std(testAccuracies), mean(testF1), std(testF1));
end;

function plot_losses(losses_train, losses_val, losses_test)
    plot(
        x_lims=(0, length(losses_train)),
        y_lims=(minimum([losses_train; losses_val; losses_test]), maximum([losses_train; losses_val; losses_test])),
        legend=:topleft,
        title="Evolución de los valores de loss",
        size=(800, 500)
    )
    plot!(losses_train, label = "Entrenamiento")
    plot!(losses_val, label = "Validación")
    plot!(losses_test, label = "Prueba")
end