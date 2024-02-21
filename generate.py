import sys, os

import yaml
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
from src.accumulator import get_accumulator
from src.datasets.tokenizer import VocabularyTokenizer

def main(config):

    result_dir = make_result_dir(**config.result_dir)
    logger = default_logger(result_dir+"/log.txt", **config.logger)
    with open(f"{result_dir}/config.yaml", 'w') as f:
        yaml.dump(config.to_dict(), f)

    # prepare models
    DEVICE = torch.device('cuda', index=config.gpuid or 0) \
        if torch.cuda.is_available() else torch.device('cpu')
    logger.warning(f"DEVICE: {DEVICE}")
    
    with open(config.voc_file) as f:
        toker = VocabularyTokenizer(f.read().splitlines())
    model_config = config.model
    model_config.update(config.model)
    model = Model(logger, **model_config)
    model.load(**config.load)
    model.to(DEVICE)
    model.eval()
    processes = [get_process(**p) for p in config.processes]
    token_accumulator = get_accumulator(logger=logger, **config.token_accumulator)
    token_accumulator.init()

    logger.info("Generating...")
    batch_size = config.batch_size
    n_generation = config.n_generation
    n_iter = (n_generation-1) // batch_size + 1
    with torch.no_grad():
        for i_iter in tqdm(range(0, n_iter)):
            if i_iter == n_iter-1: batch_size = n_generation - i_iter*batch_size
            batch = {'batch_size': batch_size}
            batch = model(batch, processes)
            token_accumulator(batch)
    token_accumulator.save(f"{result_dir}/tokens")
    tokens = token_accumulator.accums[:config.n_generation]

    logger.info("Tokenizing...")
    smiles = []
    fw = open(f"{result_dir}/smiles.txt", 'w')
    for token in tokens:
        smile = toker.detokenize(token)
        fw.write(smile+'\n')
        smiles.append(smile)
    fw.close()

if __name__ == '__main__':
    config = load_config(config_dir="./generation", default_configs=['base'])
    main(config)

