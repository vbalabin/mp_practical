import copy
import math
import numpy as np
import random as rd
from typing import List


class Layer:
    def __init__(self, size: int, next_size: int):
        self.size = size
        self.neurons = np.zeros((size,))
        self.biases = np.zeros((size,))
        self.weights = np.zeros((size, next_size))


class Point:
    x: int
    y: int
    type: int

    def __init__(self, x, y, type) -> None:
        self.x = x
        self.y = y
        self.type = type


class NeuralNetwork:
    learningRate: float
    layers: List[Layer]

    def __init__(self, learningRate, sizes) -> None:
        self.learningRate = learningRate
        self.layers = copy.copy(sizes)
        self.epoch = 0
        for i, _ in enumerate(sizes):
            next_size = 0
            if (i < len(sizes) - 1):
                next_size = sizes[i + 1]
            self.layers[i] = Layer(sizes[i], next_size)
            for j in range(sizes[i]):
                self.layers[i].biases[j] = rd.random() * 2.0 - 1.0
                for k in range(next_size):
                    self.layers[i].weights[j][k] = rd.random() * 2.0 - 1.0

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def dsigmoid(self, y):
        return y * (1 - y)

    def feed_forfard(self, inputs: List[float]):
        layers = self.layers

        for i, v in enumerate(inputs):
            layers[0].neurons[i] = v

        for i in range(1, len(layers)):
            l = layers[i-1]
            l1 = layers[i]
            for j in range(l1.size):
                l1.neurons[j] = 0
                for k in range(l.size):
                    l1.neurons[j] += l.neurons[k] * l.weights[k][j]

                l1.neurons[j] += l1.biases[j]
                l1.neurons[j] = self.sigmoid(l1.neurons[j])

        return layers[len(layers)-1].neurons

    def back_propagation(self, targets: List[float]):
        layers = self.layers
        errors = np.zeros((layers[len(layers)-1].size,))
        for i in range(layers[len(layers)-1].size):
            errors[i] = targets[i] - layers[len(layers)-1].neurons[i]

        for k in range(len(layers)-2, -1, -1):
            l = layers[k]
            l1 = layers[k+1]
            errors_next = np.zeros((l.size,))
            gradients = np.zeros((l1.size,))

            for i in range(l1.size):

                gradients[i] = errors[i] * \
                    self.dsigmoid(layers[k + 1].neurons[i])
                gradients[i] *= self.learningRate

            deltas = np.zeros((l1.size, l.size))
            for i in range(l1.size):
                for j in range(l.size):
                    deltas[i][j] = gradients[i] * l.neurons[j]

            for i in range(l.size):
                errors_next[i] = 0
                for j in range(l1.size):
                    errors_next[i] += l.weights[i][j] * errors[j]

            errors = np.zeros((l.size,))
            for i, v in enumerate(errors_next):
                errors[i] = v

            weights_new = np.zeros((len(l.weights), len(l.weights[0])))
            for i in range(l1.size):
                for j in range(l.size):
                    weights_new[j][i] = l.weights[j][i] + deltas[i][j]

            l.weights = weights_new
            for i in range(l1.size):
                l1.biases[i] += gradients[i]

        return None
