import numpy as np
import Ex1  # change to helper or sth
import matplotlib.pyplot as plt

LAYERS = [0, 1]
learningRates = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
inputsNumber = 2
inputLayerSize = 2  # hidden layer


def readData(filename, delimiter, inputsNumber):
    data = np.genfromtxt(filename, delimiter=delimiter)
    # unnecessary for XOR and AND
    normalizedData = (data - data.min()) / (data.max() - data.min())
    return [(np.asarray(item[:inputsNumber]), np.asarray(item[inputsNumber:])) for item in normalizedData]


def run(inputsOutputs, learningRate):
    desiredOutputs = [x[1] for x in inputsOutputs]
    errors = []
    for i in range(0, iterations):
        iterationOutputs = []
        for data in inputsOutputs:
            outputs = Ex1.calculateOutput(LAYERS, weights, bias, data[0])
            if len(data[1]) != len(outputs[-1]):
                raise Exception
            iterationOutputs.append(outputs[-1])
            Ex1.backpropagate(outputs, data[1], weights, data[0], bias, learningRate)
        errors.append(((1 / len(desiredOutputs)) * sum((np.asarray(desiredOutputs) - np.asarray(iterationOutputs)) ** 2))[0])
    return errors


iterations = 1000
inputsOutputs = readData('dataAND.txt', ',', inputsNumber)
errorsDict = {}
for learningRate in learningRates:
    outputLayerSize = len(inputsOutputs[0][1])
    # np.random.seed(0)
    for i in range(0,10):
        weights = np.array([np.random.rand(inputLayerSize, inputsNumber),
                            np.random.rand(outputLayerSize, inputLayerSize)])
        bias = np.array([np.random.uniform(0.0, 1.0, inputLayerSize),
                         np.random.uniform(0.0, 1.0, outputLayerSize)])

        errorsDict[i] = run(inputsOutputs, learningRate)

    for key, value in errorsDict.items():
        plt.plot(value)
    plt.title("Learning rate: " + str(learningRate))
    plt.xlabel("iteration")
    plt.ylabel("error")
    plt.show()