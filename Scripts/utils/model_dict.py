import torch.nn as nn
from collections import OrderedDict

def set_model_structure(input_shape: int, output_shape: int, *layers: tuple[dict], batch_norm: bool = True):
    """
        Function for setting up the structure for the neural network. 
        
        Parameters:
            features (int): Number of features
            *layers (tuple[int, activation_function]): Layers takes in tuples with a given number of neurons and activation function.
            batch_norm (bool):  A boolean to activate batch normalization between the layers.
    """
    activations = {
        'relu': nn.ReLU(),
        'sigmoid': nn.Sigmoid(),
        'tanh': nn.Tanh(),
        'softmax': nn.Softmax(dim=1),
        'leaky_relu': nn.LeakyReLU(negative_slope=0.01),
        'elu': nn.ELU(),
        'selu': nn.SELU(),
        'prelu': nn.PReLU(),
    }
    
    model_dict = OrderedDict()
    in_neurons = input_shape
    for i, layer in enumerate(layers, start=1):
        out_neurons, activation = layer['neurons'], layer['activation']
        model_dict['linear_layer_' + str(i)] = nn.Linear(in_neurons, out_neurons)
        if batch_norm:
            model_dict['batch_norm_'+str(i)] = nn.BatchNorm1d(out_neurons)
        
        model_dict['activation_layer_' + str(i)] = activations[activation]
        
        in_neurons = out_neurons

    output = output_shape
    if isinstance(output, tuple):
        out_neurons, activation = output
        model_dict['output_layer'] = nn.Linear(in_neurons, out_neurons)
        model_dict['output_activation_layer'] = activations[activation]
    else:
        model_dict['output_layer'] = nn.Linear(in_neurons, output)
    
    return model_dict