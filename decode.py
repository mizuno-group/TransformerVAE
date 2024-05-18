"""
LatentTransformerのfeat_evalとgenerationを統合

# TODO: ディレクトリ構造を変更する。
"""
import sys, os
os.environ.setdefault('TOOLS_DIR', "/workspace")
sys.path += [os.environ["TOOLS_DIR"]]
import yaml
import pickle
from addict import Dict
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from src.utils.path import make_result_dir
from src.utils.logger import default_logger
from src.utils.args import load_config
from src.models import Model
from src.process import get_process
from src.accumulator import get_accumulator, NumpyAccumulator, ListAccumulator
from src.metric import get_metric
from src.dataset import get_dataloader
from src.datasets.tokenizer import VocabularyTokenizer

def main(config):

    result_dir = make_result_dir(**config.result_dir)
    logger = default_logger(result_dir+"/log.txt", **config.logger)
    with open(f"{result_dir}/config.yaml", 'w') as f:
        yaml.dump(config.to_dict(), f, sort_keys=False)

    # Environment
    DEVICE = torch.device('cuda', index=config.gpuid or 0) \
        if torch.cuda.is_available() else torch.device('cpu')
    logger.warning(f"DEVICE: {DEVICE}")
    
    # Prepare data
    dl = get_dataloader(logger=logger, device=DEVICE, **config.data)

    # Prepare model
    logger.info("Preparing model...")
    model = Model(logger, **config.model)
    model.load(path=config.weight_path, strict=False)
    model.to(DEVICE)
    model.eval()
    processes = [get_process(**p) for p in config.processes]

    # Prepare hooks
    accums = {aname: get_accumulator(logger=logger, **aconfig)
        for aname, aconfig in config.accumulators.items()}
    idx_accum = NumpyAccumulator(logger=logger, input='idx', org_type='np.ndarray')
    metrics = [get_metric(logger=logger, name=mname, **mconfig) for mname, mconfig
        in config.metrics.items()]
    hooks = list(accums.values())+metrics+[idx_accum]

    for hook in hooks:
        hook.init()

    # Iteration
    logger.info("Iterating dataset...")
    with torch.no_grad():
        for batch in tqdm(dl) if config.show_tqdm else dl:
            model(batch, processes=processes)
            for hook in hooks:
                hook(batch)
            del batch
            torch.cuda.empty_cache()

    # Calculate metrics
    logger.info("Calculating metrics...")
    if len(metrics) > 0:
        scores = {}
        for m in metrics: 
            scores = m.calc(scores)
        df_score = pd.Series(scores)
        df_score.to_csv(f"{result_dir}/scores.csv", header=['Score'])
    
    # Save accumulated values
    logger.info("Saving accumulates...")
    if len(accums) > 0:
        idxs = np.argsort(idx_accum.accumulate())
        for aname, accum in accums.items():
            accummed = accum.accumulate()
            apath = f"{result_dir}/{aname}"
            if isinstance(accum, NumpyAccumulator):
                accummed = accummed[idxs]
                n, size = accummed.shape
                with open(apath+'.csv', 'w') as f:
                    f.write(','.join([str(i) for i in range(size)])+'\n')
                    for r in range(n):
                        f.write(','.join(str(f) for f in accummed[r])+'\n')
            elif isinstance(accum, ListAccumulator):
                accummed = [accummed[i] for i in idxs]
                with open(apath+'.pkl', 'wb') as f:
                    pickle.dump(accummed, f)
            else:
                raise ValueError(f"Unsupported type of accumulate: {type(accum)}")

    # Decode
    logger.info("Detokenizing...")
    with open(config.voc_file) as f:
        tokenizer = VocabularyTokenizer(f.read().splitlines())
    with open(os.path.join(result_dir, "decoded_tokens.pkl"), 'rb') as f:
        tokens = pickle.load(f)
    with open(os.path.join(result_dir, "decoded_smiles.txt"), 'w') as f:
        for tok in tokens:
            f.write(tokenizer.detokenize(tok)+'\n')

if __name__ == '__main__':
    config = load_config(config_dir="./decoding", default_configs=['base'])
    main(config)

