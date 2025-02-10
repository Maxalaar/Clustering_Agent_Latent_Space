from torch import nn


def create_dense_architecture(input_dimension, shape_layers, output_dimension, activation_function_class, layer_normalization=False, dropout=False):
    if len(shape_layers) != 0:
        layers = [nn.Linear(input_dimension, shape_layers[0]), activation_function_class()]

        for i in range(len(shape_layers) - 1):
            layers.append(nn.Linear(shape_layers[i], shape_layers[i + 1]))
            layers.append(activation_function_class())
            if layer_normalization:
                layers.append(nn.LayerNorm(shape_layers[i + 1]))
            if dropout:
                layers.append(nn.Dropout(p=0.2))

        layers.append(nn.Linear(shape_layers[-1], output_dimension, bias=False))

        return nn.Sequential(*layers)
    else:
        return nn.Linear(input_dimension, output_dimension, bias=False)
