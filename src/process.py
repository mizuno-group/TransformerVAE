from .models import function_config2func

class Process:
    def __init__(self):
        pass
    def __call__(self, model, batch):
        raise NotImplementedError

class CallProcess(Process):
    def __init__(self, input, output=None, **kwargs):
        self.input = input
        self.output = output
        if self.output is None:
            self.output = self.input
        self.kwargs = kwargs
    def __call__(self, model, batch):
        callable_ = self.get_callable(model)
        if self.input is None:
            output = callable_(**self.kwargs)
        elif isinstance(self.input, str):
            output = callable_(batch[self.input], **self.kwargs)
        elif isinstance(self.input, list):
            output = callable_(*[batch[i] for i in self.input], **self.kwargs)
        elif isinstance(self.input, dict):
            output = callable_(**{name: batch[i] for name, i in self.input.items()}, **self.kwargs)
        else:
            raise ValueError(f'Unsupported type of input: {self.input}')
        if isinstance(self.output, str):
            batch[self.output] = output
        elif isinstance(self.output, list):
            for oname, o in zip(self.output, output):
                batch[oname] = o
        else:
            raise ValueError(f'Unsupported type of output: {self.output}')
    def get_callable(self, model):
        raise NotImplementedError
class ForwardProcess(CallProcess):
    def __init__(self, module, input, output=None, **kwargs):
        """
        Parameters
        ----------
        module: str
            Name of module.
        input: str, list[str] or dict[str, str]
            Name of input(s) in the batch to the module.
        output: str, list[str], or None
            Name of output(s) in the batch from the module.
            If None, input is used as output (inplace process)
        kwargs: dict
            Other kwargs are directly send to module
        """
        super().__init__(input, output, **kwargs)
        self.module = module
    def get_callable(self, model):
        return  model[self.module]
class FunctionProcess(CallProcess):
    def __init__(self, function, input, output=None, **kwargs):
        """
        Parameters
        ----------
        function: dict
            Input for function_config2func
        input: str, list[str] or dict[str, str]
            Name of input(s) in the batch to the module.
        output: str, list[str], or None
            Name of output(s) in the batch from the module.
            If None, input is used as output (inplace process) 
        kwargs: dict
            Other kwargs are directly send to module    
        """
        super().__init__(input, output, **kwargs)
        self.function = function_config2func(function)
    def get_callable(self, model):
        return self.function

import torch
import numpy as np
class IterateProcess(Process):
    def __init__(self, length, processes, i_name='iterate_i'):
        """
        Parameters
        ----------
        length: int | str
            Length of iteration | name of it in batch.
        processes: list[dict]
            Parameters for processes to iterate
        i_name: str
            Name of index of iteration in batch        
        """
        self.length = length
        self.processes = [get_process(**process) for process in processes]
        self.i_name = i_name
    def __call__(self, model, batch):
        if isinstance(self.length, int): length = self.length
        else: length = batch[self.length]
        for i in range(length):
            batch[self.i_name] = i
            for i, process in enumerate(self.processes):
                process(model, batch)

process_type2class = {
    'forward': ForwardProcess,
    'function': FunctionProcess,
    'iterate': IterateProcess
}
def get_process(type='forward', **kwargs):
    return process_type2class[type](**kwargs)
def get_process_from_config(config):
    return get_process(**config)