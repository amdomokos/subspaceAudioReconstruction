"""STFT/ISTFT with weighted overlap-add.

N=512, H=N//4 at 16 kHz: 31 Hz/bin, 32 ms frames, 257 x ~247 matrix.
H=N//4 is the smallest hop where sum_j w[n-jH]**2 is constant (= 3K/8 = 1.5
for Hann, K=N/H=4).
"""

import numpy as np
import math
from scipy.signal import get_window



def stft(x, N=512, H=128):
    n_frames = (len(x)-N)//H +1
    stft_coefficients = np.empty((N//2 +1, n_frames ), dtype = complex)
    w = get_window("hann", N, fftbins= True) 
    for j in range(n_frames):
        frame = x[j*H : j*H + N] * w
        stft_coefficients[:, j] = np.fft.rfft(frame)
    return stft_coefficients

def istft(S, N=512, H=128):
    n_frames = S.shape[1]
    y = np.zeros((n_frames-1)*H + N)
    env = np.zeros_like(y)
    w = get_window("hann", N, fftbins=True)
    for j in range(n_frames):
        frame = np.fft.irfft(S[:, j], n=N) * w
        y[j*H : j*H + N] += frame
        env[j*H : j*H + N] += w**2 # 1.5 scaling factor only holds on interior
    return y / np.maximum(env, 1e-12)

def pad(x, N=512):
    return np.concatenate([np.zeros(N), x, np.zeros(N)])


def unpad(y, n, N=512):
    return y[N : N + n]

