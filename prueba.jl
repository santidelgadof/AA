# Función confusionMatrix que toma un vector de valores reales y opcionalmente un umbral para convertirlos en salidas binarias
function confusionMatrix(outputs::AbstractArray{<:Real, 1}, targets::AbstractArray{Bool, 1}, threshold::Real=0.5)
    outputs_binary = outputs .>= threshold
    confusionMatrix(outputs_binary, targets)
end

# Función confusionMatrix que toma un vector de valores booleanos directamente
function confusionMatrix(outputs::AbstractArray{Bool, 1}, targets::AbstractArray{Bool, 1})
    vp = sum(outputs .== true .& targets .== true)
    vn = sum(outputs .== false .& targets .== false)
    fp = sum(outputs .== true .& targets .== false)
    fn = sum(outputs .== false .& targets .== true)

    sensitivity = vp == 0 && fn == 0 ? 1.0 : vp / (fn + vp)
    precision = vp == 0 && fp == 0 ? 1.0 : vp / (vp + fp)
    specificity = vn == 0 && fp == 0 ? 1.0 : vn / (fp + vn)
    predictiveValue = vn == 0 && fn == 0 ? 1.0 : vn / (vn + fn)
    f1_score = sensitivity == 0 && precision == 0 ? 0.0 : 2 * vp / (2 * vp + fp + fn)
    
    accuracy = (vn + vp) / (vn + vp + fn + fp)
    errorRate = (fn + fp) / (vn + vp + fn + fp)

    confusion_matrix = [vp fn; fp vn]

    return (accuracy, errorRate, sensitivity, specificity, precision, predictiveValue, f1_score, confusion_matrix)
end

# Función printConfusionMatrix que toma un vector de valores booleanos
function printConfusionMatrix(outputs::AbstractArray{Bool, 1}, targets::AbstractArray{Bool, 1})
    resultados = confusionMatrix(outputs, targets)
    println("Accuracy: ", resultados[1])
    println("Error Rate: ", resultados[2])
    println("Sensitivity: ", resultados[3])
    println("Specificity: ", resultados[4])
    println("Precision: ", resultados[5])
    println("Predictive Value: ", resultados[6])
    println("F1 Score: ", resultados[7])
    println("Confusion Matrix:")
    println(resultados[8])
end

# Función printConfusionMatrix que toma un vector de valores reales y un umbral
function printConfusionMatrix(outputs::AbstractArray{<:Real, 1}, targets::AbstractArray{Bool, 1}, threshold::Real=0.5)
    resultados = confusionMatrix(outputs, targets, threshold)
    println("Accuracy: ", resultados[1])
    println("Error Rate: ", resultados[2])
    println("Sensitivity: ", resultados[3])
    println("Specificity: ", resultados[4])
    println("Precision: ", resultados[5])
    println("Predictive Value: ", resultados[6])
    println("F1 Score: ", resultados[7])
    println("Confusion Matrix:")
    println(resultados[8])
end

# Llamada a la función printConfusionMatrix con unos valores de ejemplo
outputs = [true, false, false, false]
targets = [true, true, false, false]
println("Llamada a printConfusionMatrix con valores booleanos:")
printConfusionMatrix(outputs, targets)

# Llamada a la función printConfusionMatrix con unos valores de ejemplo y un umbral
println("\nLlamada a printConfusionMatrix con valores reales y un umbral:")
outputs_reales = [0.6, 0.7, 0.2, 0.3]  # Ejemplo de salidas de un modelo con valores reales
threshold = 0.5  # Umbral para la conversión de valores reales en salidas binarias
printConfusionMatrix(outputs_reales, targets, threshold)
