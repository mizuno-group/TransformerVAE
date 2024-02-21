import sys, logging
from tqdm import tqdm
import inspect

log_name2level = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'warn': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL
}
formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s",
    "%y%m%d %H:%M:%S")

class TqdmHandler(logging.Handler):
    def __init__(self,level=logging.NOTSET):
        super().__init__(level)

    def emit(self,record):
        try:
            msg = self.format(record)
            tqdm.write(msg, file=sys.stdout)
            self.flush()
        except Exception:
            self.handleError(record)
default_loggers = {}

def default_logger(filename=None, stream_level='info', 
        file_level='debug', logger_name=None):
    if logger_name is None:
        logger_name = inspect.getmodulename(inspect.stack()[0].filename)
    logger = logging.getLogger(logger_name)
    if logger_name not in default_loggers:
        default_loggers[logger_name] = logger
        stream_handler = TqdmHandler(level=log_name2level[stream_level])
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        if filename is not None:
            file_handler = logging.FileHandler(filename=filename)
            file_handler.setLevel(log_name2level[file_level])
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)
    return logger