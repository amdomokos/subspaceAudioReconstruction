import numpy as np

def mse(ref, test):
    return np.mean((ref-test)**2)

def snr_db(ref, test):
    def rms(signal):
        return np.sqrt(np.mean(signal**2))
    return 20* np.log10(rms(ref)/rms(ref-test))

    