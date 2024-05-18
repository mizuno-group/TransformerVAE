import sys, os
import pickle
import time
import shutil
import random

import yaml
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.utils.path import make_result_dir, timestamp
from src.utils.logger import default_logger
from src.utils.args import load_config, subs_vars
from src.utils.torch import get_params
from src.dataset import get_dataloader
from src.accumulator import get_accumulator, NumpyAccumulator
from src.metric import get_metric
from src.optimizer import get_optimizer, get_scheduler
from src.process import get_process
from src.utils.utils import nullcontext
from src.hooks import AlarmHook, hook_type2class, get_hook
from src import Model
torch.autograd.set_detect_anomaly(True)

def save_rstate(dirname):
    os.makedirs(dirname, exist_ok=True)
    with open(f"{dirname}/random.pkl", 'wb') as f:
        pickle.dump(random.getstate(), f)
    with open(f"{dirname}/numpy.pkl", 'wb') as f:
        pickle.dump(np.random.get_state(), f)
    torch.save(torch.get_rng_state(), f"{dirname}/torch.pt")
    torch.save(torch.cuda.get_rng_state_all(), f"{dirname}/cuda.pt")
def set_rstate(config):
    if 'random' in config:
        with open(config.random, 'rb') as f:
            random.setstate(pickle.load(f))
    if 'numpy' in config:
        with open(config.numpy, 'rb') as f: 
            np.random.set_state(pickle.load(f))
    if 'torch' in config:
        torch.set_rng_state(torch.load(config.torch))
    if 'cuda' in config:
        torch.cuda.set_rng_state_all(torch.load(config.cuda))

def main(config, args=None):

    # make training
    ## replacement of config: add more when needed
    config = subs_vars(config, {"$TIMESTAMP": timestamp()})
    trconfig = config.training
    result_dir = make_result_dir(**trconfig.result_dir)
    logger = default_logger(result_dir+"/log.txt", trconfig.verbose.loglevel.stream, trconfig.verbose.loglevel.file)
    with open(result_dir+"/config.yaml", mode='w') as f:
        yaml.dump(config.to_dict(), f, sort_keys=False)
    if args is not None:
        logger.warning(f"options: {' '.join(args)}")

    # environment
    ## device
    DEVICE = torch.device('cuda', index=trconfig.gpuid or 0) \
        if torch.cuda.is_available() else torch.device('cpu')
    logger.warning(f"DEVICE: {DEVICE}")
    ## detect anomaly
    if trconfig.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
    ## deterministic
    if trconfig.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms = True
        torch.backends.cudnn.benchmark = False
    
    # prepare data
    dl_train = get_dataloader(logger=logger, device=DEVICE, **trconfig.data.train)
    dls_val = {name: get_dataloader(logger=logger, device=DEVICE, **dl_val_config)
        for name, dl_val_config in trconfig.data.vals.items()}
    
    # prepare model
    if 'model_seed' in trconfig:
        random.seed(trconfig.model_seed)
        np.random.seed(trconfig.model_seed)
        torch.manual_seed(trconfig.model_seed)
        torch.cuda.manual_seed(trconfig.model_seed)
    model = Model(logger=logger, **config.model)
    if trconfig.init_weight:
        model.load(**trconfig.init_weight)
    df_param, n_param, bit_size = get_params(model)
    logger.info(f"# of params: {n_param}")
    logger.info(f"Model size(bit): {bit_size}")
    model.to(DEVICE)

    optimizer = get_optimizer(params=model.parameters(), **trconfig.optimizer)
    if trconfig.optimizer_init_weight:
        optimizer.load_state_dict(torch.load(trconfig.optimizer_init_weight))
    train_processes = [get_process(**process) for process in trconfig.train_loop]
    class SchedulerAlarmHook(AlarmHook):
        def __init__(self, scheduler, **kwargs):
            super().__init__(**kwargs)
            scheduler.setdefault('last_epoch', dl_train.step - 1)
            self.scheduler = get_scheduler(optimizer, **scheduler)
        def ring(self, batch, model):
            self.scheduler.step()
    hook_type2class['scheduler_alarm'] = SchedulerAlarmHook


    # Prepare metrics
    accumulators = [ get_accumulator(logger=logger, **acc_config) for acc_config in trconfig.accumulators ]
    idx_accumulator = NumpyAccumulator(logger, input='idx', org_type='numpy')
    metrics = [ get_metric(logger=logger, name=name, **met_config) for name, met_config in trconfig.metrics.items() ]
    scores_df = pd.read_csv(trconfig.stocks.score_df, index_col="Step") \
        if trconfig.stocks.score_df else  pd.DataFrame(columns=[], dtype=float)
    val_processes = [get_process(**process) for process in trconfig.val_loop]
    if trconfig.val_loop_add_train: 
        val_processes = train_processes + val_processes
    class ValidationAlarmHook(AlarmHook):
        eval_steps = []
        def ring(self, batch, model):
            step = batch['step']
            if step in self.eval_steps: return
            self.logger.info(f"Validating step{step:7} ...")
            model.eval()
            ## evaluation
            for x in metrics+accumulators+[idx_accumulator]: x.init()
            with torch.no_grad():
                for key, dl in dls_val.items():
                    for metric in metrics:
                        metric.set_val_name(key)
                    for batch0 in dl:
                        batch0 = model(batch0, processes=val_processes)
                        for x in metrics+accumulators+[idx_accumulator]:
                            x(batch0)
                        del batch0
                        torch.cuda.empty_cache()
                    
            ## calculate & save score
            scores = {}
            for metric in metrics: scores = metric.calc(scores)
            batch.update(scores)
            for score_lb, score in scores.items():
                self.logger.info(f"  {score_lb:20}: {score:.3f}")
                scores_df.loc[step, score_lb] = score
            scores_df.to_csv(result_dir+"/val_score.csv", index_label="Step")

            ## save accumulated
            idx = idx_accumulator.accumulate()
            idx = np.argsort(idx)
            for accumulator in accumulators:
                accumulator.save(f"{result_dir}/accumulates/{accumulator.input}/{step}", indices=idx)
            
            ## save steps
            self.eval_steps.append(step)
            model.train()
    hook_type2class['validation_alarm'] = ValidationAlarmHook

    # prepare checkpoint
    class CheckpointAlarm(AlarmHook):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            os.makedirs(f"{result_dir}/checkpoints", exist_ok=True)
            self.checkpoint_steps = []
        def ring(self, batch, model: Model):
            if batch['step'] in self.checkpoint_steps: return
            checkpoint_dir = f"{result_dir}/checkpoints/{batch['step']}"
            logger.info(f"Making checkpoint at step {batch['step']:6>}...")
            if len(self.checkpoint_steps) > 0:
                shutil.rmtree(f"{result_dir}/checkpoints/{self.checkpoint_steps[-1]}/")
            os.makedirs(checkpoint_dir, exist_ok=True)
            model.save_state_dict(f"{result_dir}/models/{batch['step']}")
            torch.save(optimizer.state_dict(), f"{checkpoint_dir}/optimizer.pth")
            dl_train.checkpoint(f"{checkpoint_dir}/dataloader_train")
            scores_df.to_csv(checkpoint_dir+"/val_score.csv", index_label="Step")
            save_rstate(f"{checkpoint_dir}/rstate")
            self.checkpoint_steps.append(batch['step'])
    hook_type2class['checkpoint_alarm'] = CheckpointAlarm
    pre_hooks = [get_hook(logger=logger, result_dir=result_dir, **hconfig) for hconfig in trconfig.pre_hooks.values()]
    post_hooks = [get_hook(logger=logger, result_dir=result_dir, **hconfig) for hconfig in trconfig.post_hooks.values()]
    # load random state
    set_rstate(trconfig.rstate)

    # training
    training_start = time.time()
    logger.info("Training started.")
    with (tqdm(total=None, initial=dl_train.step) if trconfig.verbose.show_tqdm else nullcontext()) as pbar:
        now = time.time()
        optimizer.zero_grad()
        while True:

            batch = {'step': dl_train.step, 'epoch': dl_train.epoch}
            for hook in pre_hooks:
                hook(batch, model)
            
            if 'end' in batch:
                break

            # training
            batch = dl_train.get_batch(batch)
            start = time.time()
            batch = model(batch, processes=train_processes)
            loss = sum(batch[loss_name] for loss_name in trconfig.loss_names)
            if trconfig.regularize_loss.normalize:
                loss = loss / loss.detach()
            loss.backward()

            if trconfig.regularize_loss.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 
                max_norm=trconfig.regularize_loss.clip_grad_norm, error_if_nonfinite=True)
            if trconfig.regularize_loss.clip_grad_value:
                torch.nn.utils.clip_grad_value_(model.parameters(),
                clip_value=trconfig.regularize_loss.clip_grad_value)
            if dl_train.step % trconfig.schedule.opt_freq == 0:
                optimizer.step()
                optimizer.zero_grad()
            batch['time'] = time.time() - start
                
            for hook in post_hooks:
                hook(batch, model)
            del batch, loss
            torch.cuda.empty_cache()
            if pbar is not None: pbar.update(1)
    for hook in pre_hooks+post_hooks:
        hook(batch, model)

if __name__ == '__main__':
    config = load_config("training", default_configs=['base'])
    main(config, sys.argv)
