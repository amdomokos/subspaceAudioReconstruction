import numpy as np
from scipy.signal import get_window
from src.transform import stft, istft, pad, unpad

N, H = 512, 128


def test_roundtrip_is_exact():
    """The gate: analysis then synthesis must return the signal."""
    rng = np.random.default_rng(0)
    for x in [rng.standard_normal(16000),
              np.sin(2*np.pi*440*np.arange(16000)/16000),
              rng.standard_normal(4096)]:
        y = unpad(istft(stft(pad(x))), len(x))
        assert np.max(np.abs(x - y)) < 1e-10


def test_shape_dtype_and_bin_mapping():
    """Catches a transposed matrix or a dropped imaginary part."""
    fs = 16000
    x = np.sin(2*np.pi*440*np.arange(fs)/fs)
    S = stft(x)
    assert S.shape == (N//2 + 1, (len(x) - N)//H + 1)
    assert S.dtype == np.complex128
    assert np.abs(S).argmax() // S.shape[1] == round(440*N/fs)   # bin 14


def test_cola_envelope_is_flat():
    """Guards the N, H choice: sum of w**2 must be constant in the interior."""
    w = get_window("hann", N, fftbins=True)
    env = np.zeros(60*H + N)
    for j in range(60):
        env[j*H : j*H + N] += w**2
    assert np.allclose(env[N:-N], 1.5)