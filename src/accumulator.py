import os
import pickle

import numpy as np

from .utils.utils import check_leftargs, EMPTY

class NumpyAccumulator:
    def __init__(self, logger, input, batch_dim=0, org_type='torch.tensor', **kwargs):
        """
        Parameters
        ----------
        input: [str] Key of value in batch to accumulate
        org_type: [str, 'list', 'torch.tensor', 'np.array'; default='torch.tensor']
            Type of value to accumulate
        batch_dim: [int; default=0] Dimension of batch
        """
        check_leftargs(self, logger, kwargs)
        self.input = input
        if org_type in {'tensor', 'torch', 'torch.tensor'}:
            self.converter = lambda x: x.cpu().numpy()
        elif org_type in {'np.array', 'np.ndarray', 'numpy', 'numpy.array', 'numpy.ndarray'}:
            self.converter = EMPTY
        else:
            raise ValueError(f"Unsupported type of config.org_type: {org_type} in NumpyAccumulator")
        self.batch_dim = batch_dim
    def init(self):
        self.accums = []
    def accumulate(self, indices=None):
        accums = np.concatenate(self.accums, axis=self.batch_dim)
        if indices is not None:
            accums = accums[indices]
        return accums
    def save(self, path_without_ext, indices=None):
        path = path_without_ext + ".npy"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, self.accumulate(indices=indices))
    def __call__(self, batch):
        self.accums.append(self.converter(batch[self.input]))
class ListAccumulator:
    def __init__(self, logger, input, org_type='torch.tensor', batch_dim=None, **kwargs):
        """
        Parameters
        ----------
        input: [str] Key of value in batch to accumulate
        org_type: [str, 'list', 'torch.tensor', 'np.array'] Type of value
                    to accumulate
        """
        check_leftargs(self, logger, kwargs)
        self.input = input
        if org_type == 'list':
            assert batch_dim is None, f"batch_dim cannot be defined when org_type is list"
            self.converter = EMPTY
        else:
            if batch_dim is None: batch_dim = 0
            if org_type in {'tensor', 'torch.tensor'}:
                if batch_dim == 0:
                    self.converter = lambda x: list(x.cpu().numpy())
                else:
                    self.converter = lambda x: list(x.transpose(batch_dim, 0).cpu().numpy())
            elif org_type in {'np.array', 'np.ndarray', 'numpy', 'numpy.array', 'numpy.ndarray'}:
                if batch_dim == 0:
                    self.converter = lambda x: list(x)
                else:
                    self.converter = lambda x: list(x.swapaxes(0, batch_dim))
    def init(self):
        self.accums = []
    def accumulate(self, indices=None):
        if indices is not None:
            accums = np.array(self.accums, dtype=object)
            return accums[indices].tolist()
        else:
            return self.accums
    def save(self, path_without_ext, indices=None):
        path = path_without_ext + '.pkl'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(f"{path_without_ext}.pkl", 'wb') as f:
            pickle.dump(self.accumulate(indices=indices), f)
    def __call__(self, batch):
        self.accums += self.converter(batch[self.input])

accumulator_type2class = {
    'numpy': NumpyAccumulator,
    'list': ListAccumulator
}
def get_accumulator(type, **kwargs):
    return accumulator_type2class[type](**kwargs)