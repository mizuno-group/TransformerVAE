from contextlib import contextmanager

@contextmanager
def nullcontext():
    yield None

def prog(marker='*'):
    print(marker, flush=True, end='')

def check_leftargs(self, logger, kwargs, show_content=False):
    if len(kwargs) > 0 and logger is not None:
        logger.warning(f"Unknown kwarg in {type(self).__name__}: {kwargs if show_content else list(kwargs.keys())}")

EMPTY = lambda x: x
