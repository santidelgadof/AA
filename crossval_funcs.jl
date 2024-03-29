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

function modelCrossValidation(modelType::Symbol, modelHyperparameters::Dict, inputs::AbstractArray{<:Real,2}, targets::AbstractArray{<:Any,1}, crossValidationIndices::Array{Int64,1})
    @assert(size(inputs,1)==length(targets));
    classes = unique(targets);

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
                        testDataset =       (testInputs, testTargets);
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