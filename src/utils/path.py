import sys, os
import re
import shutil
from glob import glob
import subprocess
import pickle
from datetime import datetime
import pandas as pd

def cleardir(dirname, exist_ok=None):
    _cleardir(dirname)
    os.makedirs(dirname)

def _cleardir(dirname):
    for path in glob(os.path.join(dirname, '*')):
        if os.path.isdir(path):
            _cleardir(path)
        else:
            os.remove(path)
    if os.path.exists(dirname):
        os.rmdir(dirname)

def make_result_dir(dirname=None, duplicate=None):
    if os.path.exists(dirname):
        if duplicate == 'error':
            raise FileExistsError(f"'{dirname}' already exists.")
        elif duplicate == 'ask':
            answer = None
            while answer not in ['y', 'n']:
                answer = input(f"'{dirname}' already exists. Will you overwrite? (y/n)")
            if answer == 'n':
                return
        elif duplicate in {'overwrite', 'merge'}:
            pass
        else:
            raise ValueError(f"Unsupported config.result_dir.duplicate: {duplicate}")
    if duplicate == 'merge':
        os.makedirs(dirname, exist_ok=True)
    else:
        cleardir(dirname)
    return dirname

def timestamp():
    dt_now = datetime.now()
    return f"{dt_now.year%100:02}{dt_now.month:02}{dt_now.day:02}"