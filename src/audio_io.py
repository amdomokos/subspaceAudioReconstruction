import math
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

def preprocess(x, fs, target_fs=16000):
    x = np.asarray(x, dtype=float)
    if x.ndim == 2:
        x = x[:, 0]
    x =  x - np.mean(x)
    x = resample_poly(x, target_fs, fs)
    mask = np.abs(x) > 1e-4 * np.abs(x).max()
    idx = np.flatnonzero(mask)
    x = x[idx[0] : idx[-1] + 1]
    scaling_factor = np.abs(x).max()/0.8
    x = x/scaling_factor
    return x, target_fs