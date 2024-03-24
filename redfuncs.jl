
function buildANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int, transferFunctions::AbstractArray{<:Function,1})
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

function trainANN(topology::AbstractVector{<:Int64}, dataset::Tuple{AbstractMatrix{<:Float32}, AbstractMatrix{Bool}},
    transferFunctions::AbstractVector{<:Function}, maxEpochs::Int64=1000, minLoss::Real=0.0, learningRate::Real=0.01)

    (inputs, targets) = dataset;

    # Creamos la RNA
    ann = buildANN(size(inputs,2), topology, size(targets,2), transferFunctions);

    # Definimos la funcion de loss
    loss(x,y) = (size(y,1) == 1) ? Flux.Losses.binarycrossentropy(ann(x),y) : Flux.Losses.crossentropy(ann(x),y);

    # Creamos los vectores con los valores de loss y de precision en cada ciclo
    trainingLosses = Float32[];

    # Empezamos en el ciclo 0
    numEpoch = 0;
    # Calculamos el loss para el ciclo 0 (sin entrenar nada)
    trainingLoss = loss(inputs', targets');
    #  almacenamos el valor de loss y precision en este ciclo
    push!(trainingLosses, trainingLoss);
    #  y lo mostramos por pantalla
    println("Epoch ", numEpoch, ": loss: ", trainingLoss);

    # Entrenamos hasta que se cumpla una condicion de parada
    while (numEpoch<maxEpochs) && (trainingLoss>minLoss)

        # Entrenamos 1 ciclo. Para ello hay que pasar las matrices traspuestas (cada patron en una columna)
        Flux.train!(loss, Flux.params(ann), [(inputs', targets')], ADAM(learningRate));

        # Aumentamos el numero de ciclo en 1
        numEpoch += 1;
        # Calculamos las metricas en este ciclo
        trainingLoss = loss(inputs', targets');
        #  almacenamos el valor de loss
        push!(trainingLosses, trainingLoss);
        
        println("Epoch ", numEpoch, ": loss: ", trainingLoss);

    end;

    return (ann, trainingLosses);
end;