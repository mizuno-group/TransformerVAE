import torch
import torch.nn as nn
from ..models import init_config2func, function_config2func

class Affine(nn.Module):
    def __init__(self, weight, bias, input_size):
        super().__init__()
        self.weight = nn.Parameter(torch.full((input_size,), fill_value=float(weight)))
        self.bias = nn.Parameter(torch.full((input_size,), fill_value=float(bias)))
    def forward(self, input):
        return input*self.weight+self.bias
class BatchSecondBatchNorm(nn.Module):
    def __init__(self, input_size, args):
        super().__init__()
        self.norm = nn.BatchNorm1d(num_features=input_size, **args)
    def forward(self, input):
        input = input.transpose(0, 1)
        input = self.norm(input)
        return input.transpose()
    
def get_layer(config, input_size):    
    if config.type == 'view':
        new_shape = []
        for size in config.shape:
            if size == 'batch_size':
                assert -1 not in new_shape, f"Invalid config.shape: {config.shape}"
                size = -1
            new_shape.append(size)
        layer = lambda x: x.view(*new_shape)
        input_size = config.shape
    elif config.type == 'slice':
        slices = (slice(*slice0) for slice0 in config.slices)
        layer = lambda x: x[slices]
        for dim, slice0 in enumerate(slices):
            if isinstance(input_size[dim], int):
                start = slice0.start if slice0.start is not None else 0
                stop = slice0.stop if slice0.start is not None else input_size[dim]
                step = slice0.step if slice0.stop is not None else 1
                input_size[dim] = (stop - start) // step
    elif config.type == 'squeeze':
        config.setdefault('dim', None)
        layer = lambda x: torch.squeeze(x, dim=config.dim)
        if config.dim == None:
            input_size = [s for s in input_size if s != 1]
        else:
            if input_size[config.dim] != 1:
                raise ValueError(f"{config.dim} th dim of size {input_size} is not squeezable.")
            size = list(input_size)[:config.dim]+list(input_size[config.dim+1:])
        input_size = size
    elif config.type in ['norm', 'layernorm', 'ln']:
        layer = nn.LayerNorm(input_size[-1], **config.args)
        init_config2func(config.init.weight)(layer.weight)
        init_config2func(config.init.bias)(layer.bias)
    elif config.type in ['batchnorm', 'bn']:
        layer = nn.BatchNorm1d(input_size[-1], **config.args)
        init_config2func(config.init.weight)(layer.weight)
        init_config2func(config.init.bias)(layer.bias)
    elif config.type in ['batchsecond_batchnorm', 'bsbn']:
        layer = BatchSecondBatchNorm(input_size[-2], args=config.args)
    elif config.type == "linear":
        layer = nn.Linear(input_size[-1], config.size, **config.args)
        init_config2func(config.init.weight)(layer.weight)
        init_config2func(config.init.bias)(layer.bias)
        input_size = input_size[:-1]+[config.size]
    elif config.type in ["laffine", "affine"]:
        layer = Affine(config.init.weight, config.init.bias, input_size[-1])
    elif config.type == "function":
        layer = function_config2func(config.function)
    elif config.type == "dropout":
        layer = nn.Dropout(**config.args)
    else:
        raise ValueError(f"Unsupported config: {config.type}")
    return layer, input_size
class Layer(nn.Module):
    def __init__(self, layer, input_size):
        super().__init__()
        self.layer, _ = get_layer(layer, input_size)
    def forward(self, input):
        return self.layer(input)
class Tunnel(nn.Module):
    def __init__(self, layers, input_size):
        super().__init__()
        self.layers = []
        modules = []
        for i_layer, layer_config in enumerate(layers):
            layer, input_size = get_layer(layer_config, input_size)
            self.layers.append(layer)
            if isinstance(layer, nn.Module):
                modules.append(layer)
        self.modules_ = nn.ModuleList(modules)
    def forward(self, input):
        next_input = input
        for layer in self.layers:
            next_input = layer(next_input)
        return next_input