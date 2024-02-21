import os
import time
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from .alarm import get_alarm

class AlarmHook:
    def __init__(self, logger, result_dir, 
        alarm={'type': 'silent', 'target': 'step'}, end=False):
        self.logger = logger
        if not isinstance(alarm, list):
            alarm = [alarm]
        self.alarms = [get_alarm(logger=logger, **a) for a in alarm]
        self.end = end
    def __call__(self, batch, model):
        ring = False
        for alarm in self.alarms:
            if alarm(batch):
                ring = True
        if ring or ('end' in batch and self.end):
            self.ring(batch=batch, model=model)
    def ring(self, batch, model):
        raise NotImplementedError

class SaveAlarmHook(AlarmHook):
    def __init__(self, logger, result_dir, 
        alarm={'type': 'silent', 'target': 'step'}, end=False):
        super().__init__(logger, result_dir, alarm, end=end)
        self.models_dir = f"{result_dir}/models"
        os.makedirs(self.models_dir, exist_ok=True)
    def ring(self, batch, model):
        self.logger.info(f"Saving model at step {batch['step']:2>}...")
        path = f"{self.models_dir}/{batch['step']}"
        model.save_state_dict(path)

class AccumulateHook:
    def __init__(self, logger, result_dir, names, cols, save_alarm,
            checkpoint=None, fname='steps'):
        os.makedirs(result_dir, exist_ok=True)
        self.path_df = result_dir+f"/{fname}.csv"
        self.save_alarm = get_alarm(logger=logger, **save_alarm)
        self.dfs = []
        if checkpoint is not None:
            self.dfs.append(pd.read_csv(checkpoint))
        self.lists = defaultdict(list)
        self.names = names
        self.cols = cols

    def __call__(self, batch, model):
        if 'end' not in batch:
            for name, col in zip(self.names, self.cols):
                item = batch[name]
                if isinstance(item, torch.Tensor):
                    item = item.detach().cpu().numpy()
                self.lists[col].append(item)        
        if self.save_alarm(batch):
            self.dfs.append(pd.DataFrame(self.lists))
            self.lists.clear()
            pd.concat(self.dfs).to_csv(self.path_df, index=False)

class AbortHook:
    def __init__(self, logger, result_dir, 
        target, threshold):
        """
        Parameters
        ----------
        target: str
            Target value in batch
        threshold: int or float
            When batch[target] >= threshold, batch['end'] is added.
        """
        self.target = target
        self.threshold = threshold
    def __call__(self, batch, model):
        if self.target in batch and \
            batch[self.target] >= self.threshold:
            batch['end'] = True
class StepAbortHook(AbortHook):
    def __init__(self, logger, result_dir, threshold):
        super().__init__(logger, result_dir, 'step', threshold)
class EpochAbortHook(AbortHook):
    def __init__(self, logger, result_dir, threshold):
        super().__init__(logger, result_dir, 'epoch', threshold)
class TimeAbortHook:
    def __init__(self, logger, result_dir, 
        threshold):
        """
        Parameters
        ----------
        threshold: int
            Second to continue        
        """
        self.end = time.time() + threshold
    def __call__(self, batch, model):
        if time.time() > self.end:
            batch['end'] = True

hook_type2class = {
    'save_alarm': SaveAlarmHook,
    'accumulate': AccumulateHook,
    'abort': AbortHook,
    'step_abort': StepAbortHook,
    'epoch_abort': EpochAbortHook,
    'time_abort': TimeAbortHook
}
def get_hook(type, **kwargs):
    return hook_type2class[type](**kwargs)