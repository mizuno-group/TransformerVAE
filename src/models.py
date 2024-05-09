import os
import math
import logging
from collections import OrderedDict
from inspect import signature
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

# functions
def NewGELU(input):
    return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) \
        * (input + 0.044715 * torch.pow(input, 3.0))))
def sigmoid(input):
    return 1/(1+math.e**(-input))

init_type2func = {
    'glorot_uniform': nn.init.xavier_uniform_,
    'glorot_normal': nn.init.xavier_normal_,
    'he_uniform': nn.init.kaiming_uniform_,
    'he_normal': nn.init.kaiming_normal_,
    'zero': nn.init.zeros_,
    'zeros': nn.init.zeros_,
    'one': nn.init.ones_,
    'ones': nn.init.ones_,
    'normal': nn.init.normal_,
    'none': lambda input: None,
}


def init_config2func(type='none', factor=None, **kwargs):
    """
    Parameters
    ----------
    type: int, float, str or dict  
    """
    
    if factor is not None:
        def init(input: nn.Parameter):
            init_config2func(type, **kwargs)(input)
            input.data = input.data * factor
        return init

    if isinstance(type, dict):
        return init_config2func(**type)
    
    if isinstance(type, (int, float)):
        return lambda input: nn.init.constant_(input, float(type))
    elif type in init_type2func:
        return lambda input: init_type2func[type](input, **kwargs)
    else:
        raise ValueError(f"Unsupported type of init function: {type}")

function_name2func = {
    'relu': F.relu,
    'gelu': F.gelu,
    'sigmoid': torch.sigmoid,
    'tanh': torch.tanh,
    'newgelu': NewGELU,
    'none': lambda input: input,
    'exp': torch.exp,
    'log': torch.log,
    'sum': torch.sum,
    'mean': torch.mean,
    'log_softmax': F.log_softmax,
    'softplus': F.softplus,
    'transpose': torch.transpose,
    'argmax': torch.argmax
}
def function_config2func(config):
    if isinstance(config, str):
        return function_name2func[config]
    else:
        return partial(function_name2func[config.pop('type')], **config)

# Modules
module_type2class = {}
class Affine(nn.Module):
    def __init__(self, weight=1.0, bias=0.0):
        super().__init__()
        self.weight = weight
        self.bias = bias
    def forward(self, input):
        return input*self.weight+self.bias
for cls in [Affine]:
    module_type2class[cls.__name__] = cls

# Model
def get_module(logger, type, **kwargs):
    cls = module_type2class[type]
    args = set(signature(cls.__init__).parameters.keys())
    uargs = {}
    for key in list(kwargs.keys()):
        if key in args:
            uargs[key] = kwargs.pop(key)
    if len(kwargs) > 0:
        logger.warning(f"Unknown kwarg in {cls.__name__}: {kwargs}")
    return cls(**uargs)
class Model(nn.ModuleDict):
    def __init__(self, logger: logging.Logger, modules: dict, use_modules:list=None,
            omit_modules: list=None, seed=None, init: dict = {}):
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        if (use_modules is not None) and (omit_modules is not None):
            raise ValueError(f"Please specify either use_modules or omit_modules")
        mods = OrderedDict()
        for mod_name, mod_config in modules.items():
            if (use_modules is not None and mod_name not in use_modules) or \
                (omit_modules is not None and mod_name in omit_modules):
                continue
            logger.debug(f"Building {mod_name}...")
            mods[mod_name] = get_module(logger=logger, **mod_config)
        logger.debug("Building finished.")
        super().__init__(modules=mods)
        self.logger = logger

        # initialization
        for name, param in self.state_dict().items():
            for pattern, config in init.items():
                if pattern in name:
                    init_config2func(config)(param)

    def forward(self, batch, processes: list):
        for process in processes:
            process(self, batch)
        return batch
    def load(self, path, replace={}, strict=True):
        if os.path.isfile(path):
            state_dict = torch.load(path)
            for key, replace in replace.items():
                for sdict_key, sdict_value in list(state_dict.items()):
                    if sdict_key[:len(key)] == key:
                        state_dict[(replace+sdict_key[len(key):])] = sdict_value
                        del state_dict[sdict_key]
            keys = self.load_state_dict(state_dict, strict=strict)
            if len(keys.missing_keys) > 0:
                self.logger.warning("Missing keys: ")
                for key in keys.missing_keys:
                    self.logger.warning(f"  {key}")
            if len(keys.unexpected_keys) > 0:
                self.logger.warning("Unexpected keys: ")
                for key in keys.unexpected_keys:
                    self.logger.warning(f"  {key}")
        elif os.path.isdir(path):
            replace_inverse = {value: key for key, value in replace.items()}
            for mname, module in self.items():
                if mname in replace_inverse:
                    mname = replace_inverse[mname]
                mpath = f"{path}/{mname}.pth"
                if os.path.exists(mpath):
                    keys = module.load_state_dict(torch.load(mpath), strict=strict)
                    if len(keys.missing_keys) > 0:
                        self.logger.warning(f"Missing keys in {mname}: ")
                        for key in keys.missing_keys:
                            self.logger.warning(f"  {key}")
                    if len(keys.unexpected_keys) > 0:
                        self.logger.warning(f"Unexpected keys in {mname}: ")
                        for key in keys.missing_keys:
                            self.logger.warning(f"  {key}")
                else:
                    if strict:
                        raise ValueError(f"State dict file of {mname} does not exists.")
                    else:
                        self.logger.warning(f"State dict file of {mname} does not exists.")
        else:
            if os.path.exists(path):
                raise ValueError(f"Invalid file: {path}")
            else:
                raise FileNotFoundError(f"No such file or directory: {path}")    
    def save_state_dict(self, path):
        os.makedirs(path, exist_ok=True)
        for key, module in self.items():
            torch.save(module.state_dict(), os.path.join(path, f"{key}.pth"))
