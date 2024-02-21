import torch
from torch.optim import lr_scheduler

optimizer_type2class = {
    'adam': torch.optim.Adam,
    'adamw': torch.optim.AdamW,
}
def get_optimizer(type, **kwargs):
    return optimizer_type2class[type](**kwargs)

scheduler_type2class = {
    'multistep': lr_scheduler.MultiStepLR,
    'linear': lr_scheduler.LinearLR,
    'exponential': lr_scheduler.ExponentialLR
}
def get_scheduler(optimizer, type, last_epoch=-1, **kwargs):
    if type in scheduler_type2class:
        return scheduler_type2class[type](optimizer=optimizer, **kwargs)
    else:
        if type == 'warmup':
            warmup_step = kwargs['warmup']
            schedule = lambda step: min((warmup_step/(step+1))**0.5, (step+1)/warmup_step)
        elif type == 'reciprocal':
            warmup_step = kwargs['warmup']
            schedule = lambda step: ((step+1)/warmup_step)**-0.5
        elif type == 'noam':
            # for old Transformer
            factor = kwargs['d_model']**-0.5
            wfactor = kwargs['warmup']**-1.5
            schedule = lambda step: factor*min((step+1)**-0.5, (step+1)*wfactor)
        else:
            raise ValueError(f"Unsupported type of scheduler: {type}")
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=schedule, last_epoch=last_epoch)