import numpy as np


def sigmoid(activation):
    return 1.0 / (1.0 + np.exp(-activation))


def sigmoidDerivative(output):
    return output * (1.0 - output)


def getError(output, expected):
    return (expected - output) * sigmoidDerivative(output)


def getLayerOutput(inputs, weights, bias):
    if weights.shape[1] % len(inputs):  # number of columns same as inputs number
        raise Exception
    sumations = weights @ inputs  # matrices multiplication
    sumations = sumations + bias  # element wise list adding
    return [sigmoid(x) for x in sumations]


def calculateOutput(layers, weights, bias, inputs):
    layersOutputs = []
    for layer in layers:
        inputs = np.asarray(getLayerOutput(inputs, weights[layer], bias[layer]))
        layersOutputs.append(inputs)
    return np.asarray(layersOutputs)


def updateWeights(weights, updates):
    for idx, weightsLayer in enumerate(weights):
        weights[idx] = weightsLayer + updates[idx]


def getErrors(outputs, desired):
    error = []
    for idx, x in enumerate(desired - outputs):
        error.append(x * sigmoidDerivative(outputs[idx]))
    return error


def backpropagate(outputs, desiredNetworkOutputs, weights, inputs, bias, learningRate):
    # NN outputs error, last layer
    errorsOutputLayer = np.asarray(getErrors(outputs[-1], desiredNetworkOutputs))
    errorsArrayTranposed = np.array([errorsOutputLayer]).T
    costFunctionUpdatesMatrixLayer3 = errorsArrayTranposed * outputs[-2]  # convert output errors to array 2dim to transpose
    errorsHiddenLayer = errorsOutputLayer @ weights[-1] * np.asarray([sigmoidDerivative(hiddenOutput) for hiddenOutput in outputs[-2]])
    costFunctionUpdatesMatrixLayer2 = np.array([errorsHiddenLayer]).T @ np.array([inputs])
    updateWeights(weights, np.array([costFunctionUpdatesMatrixLayer2 * learningRate,
                                     costFunctionUpdatesMatrixLayer3 * learningRate]))
    updateWeights(bias, np.array([errorsHiddenLayer * learningRate,
                                  errorsOutputLayer * learningRate]))
