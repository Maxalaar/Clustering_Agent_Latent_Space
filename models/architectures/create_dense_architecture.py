from torch import nn


def create_dense_architecture(input_dimension, shape_layers, output_dimension, activation_function):
    layers = [nn.Linear(input_dimension, shape_layers[0]), activation_function]

    for i in range(len(shape_layers) - 1):
        layers.append(nn.Linear(shape_layers[i], shape_layers[i + 1]))
        layers.append(activation_function)

    layers.append(nn.Linear(shape_layers[-1], output_dimension))

    return nn.Sequential(*layers)