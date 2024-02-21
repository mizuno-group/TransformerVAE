import pandas as pd
import torch

def get_device(logger=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if logger is not None:
        logger.warning(f"DEVICE: {device}")
    return device

def get_params(model: torch.nn.Module):

    names = []
    types = []
    shapes = []
    n_param = 0
    bit_size = 0
    for key, value in model.state_dict().items():
        names.append(key)
        types.append(value.dtype)
        shapes.append(','.join([str(s) for s in value.shape]))
        n_param0 = 1
        for s in value.shape: n_param0 *= s
        n_param += n_param0
        bit_size += torch.finfo(value.dtype).bits*n_param0

    df = pd.DataFrame({
        'name': names,
        'type': types,
        'shape': shapes
    })
    return df, n_param, bit_size
