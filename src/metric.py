from collections import defaultdict
import numpy as np
import torch

from .utils.utils import check_leftargs
from sklearn.metrics import roc_auc_score, average_precision_score, \
    mean_squared_error, mean_absolute_error, r2_score

class Metric:
    def __init__(self, logger, name, **kwargs):
        check_leftargs(self, logger, kwargs)
        self.val_name = ''
        self.name = name
    def set_val_name(self, val_name):
        self.val_name = val_name
    def init(self):
        raise NotImplementedError
    def add(self, batch):
        raise NotImplementedError
    def calc(self):
        raise NotImplementedError
    def __call__(self, batch):
        self.add(batch)

class BinaryMetric(Metric):
    def __init__(self, logger, name, input, target, is_logit, is_multitask=False, 
            input_process=None, task_names=None, **kwargs):
        """
        Parameters
        ----------
        is_logit: bool
            If True, batch[input] is used as decision function.
            If False, batch[input][:, 1] is used as decision function. 
        is_multitask: bool
            If True, decision function should be [batch_size, n_task(, 2)]
            If False, decision function should be [batch_size(, 2)]
        input_process: str or None
            None: nothing applied to decision function
            'softmax': softmax function is applied (is_logit must be False)
            'sigmoid': sigmoid function is applied (is_logit must be True)
        task_names: List[str] or None
        """
        super().__init__(logger, name, **kwargs)
        self.input = input
        self.target = target
        self.is_logit = bool(is_logit)
        assert input_process is None or input_process in {'softmax', 'sigmoid'}
        if input_process == 'softmax': assert not is_logit
        elif input_process == 'sigmoid': assert is_logit
        self.is_multitask = is_multitask
        self.input_process = input_process
        self.task_names = task_names
    def init(self):
        self.targets = defaultdict(list)
        self.inputs = defaultdict(list)
    def add(self, batch):
        self.targets[self.val_name].append(batch[self.target].cpu().numpy())
        input = batch[self.input]
        if self.input_process == 'softmax':
            input = torch.softmax(input, dim=-1)
        elif self.input_process == 'sigmoid':
            input = torch.sigmoid(input)
        input = input.cpu().numpy()
        if not self.is_logit:
            input = input[..., 1]
        self.inputs[self.val_name].append(input)
    def calc(self, scores):
        total_inputs = []
        total_targets = []
        for val_name in self.targets.keys():
            input = np.concatenate(self.inputs[val_name])
            target = np.concatenate(self.targets[val_name])
            if len(self.targets) > 1:
                if self.is_multitask:
                    task_names = self.task_names if self.task_names is not None else range(target.shape[1]) 
                    for i_task, task_name in enumerate(task_names):
                        scores[f"{val_name}_{task_name}_{self.name}"] = \
                            self.calc_score(y_true=target[:,i_task], y_score=input[:, i_task])
                else:
                    scores[f"{val_name}_{self.name}"] = self.calc_score(y_true=target, y_score=input)
            total_inputs.append(input)
            total_targets.append(target)
        if len(total_inputs) == 0: return scores
        total_inputs = np.concatenate(total_inputs, axis=0)
        total_targets = np.concatenate(total_targets, axis=0)
        if self.is_multitask:
            for i_task, task_name in zip(range(target.shape[1]), self.task_names):
                scores[f"{task_name}_{self.name}"] = \
                    self.calc_score(y_true=total_targets[:,i_task], y_score=total_inputs[:, i_task])
        else:
            scores[f"{self.name}"] = \
                self.calc_score(y_true=total_targets, y_score=total_inputs)
        return scores
    def calc_score(self, y_true, y_score):
        raise NotImplementedError
        
class AUROCMetric(BinaryMetric):
    def calc_score(self, y_true, y_score):
        if np.all(y_true == y_true[0]):
            return 0
        else:
            return roc_auc_score(y_true=y_true, y_score=y_score)
class AUPRMetric(BinaryMetric):
    def calc_score(self, y_true, y_score):
        if np.all(y_true == y_true[0]):
            return 0
        else:
            return average_precision_score(y_true=y_true, y_score=y_score)
class RMSEMetric(BinaryMetric):
    def calc_score(self, y_true, y_score):
        return np.sqrt(mean_squared_error(y_true=y_true, y_pred=y_score))
class MAEMetric(BinaryMetric):
    def calc_score(self, y_true, y_score):
        return mean_absolute_error(y_true=y_true, y_pred=y_score)
class R2Metric(BinaryMetric):
    def calc_score(self, y_true, y_score):
        return r2_score(y_true=y_true, y_pred=y_score)

class MeanMetric(Metric):
    def init(self):
        self.scores = defaultdict(list)
    def calc(self, scores):
        total_values = []
        for val_name, values in self.scores.items():
            if values[0].ndim == 0:
                values = np.array(values)
            else:
                values = np.concatenate(values)
            if len(self.scores) > 1:
                scores[f"{val_name}_{self.name}"] = np.mean(values)
            total_values.append(values)
        if len(self.scores) > 0:
            scores[self.name] = np.mean(np.concatenate(total_values))
        return scores
class ValueMetric(MeanMetric):
    def add(self, batch):
        self.scores[self.val_name].append(batch[self.name].cpu().numpy())
class PerfectAccuracyMetric(MeanMetric):
    def __init__(self, logger, name, input, target, pad_token, **kwargs):
        super().__init__(logger, name, **kwargs)
        self.name = name
        self.input = input
        self.target = target
        self.pad_token = pad_token
    def add(self, batch):
        self.scores[self.val_name].append(torch.all((batch[self.input] == batch[self.target])
            ^(batch[self.target] == self.pad_token), axis=1).cpu().numpy())
class PartialAccuracyMetric(MeanMetric):
    def __init__(self, logger, name, input, target, pad_token, **kwargs):
        super().__init__(logger, name, **kwargs)
        self.name = name
        self.input = input
        self.target = target
        self.pad_token = pad_token
    def add(self, batch):
        target_seq = batch[self.target]
        pred_seq = batch[self.input]
        pad_mask = (target_seq != self.pad_token).to(torch.int)
        self.scores[self.val_name].append((torch.sum((target_seq == pred_seq)*pad_mask, dim=1)
            /torch.sum(pad_mask, dim=1)).cpu().numpy())
metric_type2class = {
    'value': ValueMetric,
    'auroc': AUROCMetric,
    'aupr': AUPRMetric,
    'rmse': RMSEMetric,
    'mae': MAEMetric,
    'r2': R2Metric,
    'perfect': PerfectAccuracyMetric,
    'partial': PartialAccuracyMetric,
}
def get_metric(type, **kwargs) -> Metric:
    return metric_type2class[type](**kwargs)