import numpy as np
from .eps import eps

def random_shift(x, fraction):
    """Apply a random shift to a pandas series."""
    min_x, max_x = np.min(x), np.max(x)
    m = np.random.uniform(-fraction, fraction, size=x.shape) + 1
    return np.clip(x * m, min_x, max_x)


def normalize(x):
    """Normalize to a pandas series."""
    x = (x - x.mean()) / (x.std() + eps)
    return x


def scale_to_start(x):
    """Scale pandas series so that it starts at one."""
    x = (x + eps) / (x[0] + eps)
    return x

def softmax(w, t=1.0):
    """softmax implemented in numpy."""
    log_eps = np.log(eps)
    w = np.clip(w, log_eps, -log_eps)  # avoid inf/nan
    e = np.exp(np.array(w) / t)
    dist = e / np.sum(e)
    return dist