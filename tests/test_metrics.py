import numpy as np
from src.metrics import snr_db

def test_zeros_is_exactly_zero_db():
    x = np.random.default_rng(0).standard_normal(100)
    assert snr_db(x, np.zeros_like(x)) == 0.0

def test_matches_power_form():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(500)
    y = x + 0.05 * rng.standard_normal(500)
    expected = 10 * np.log10(np.mean(x**2) / np.mean((x - y) ** 2))
    assert np.isclose(snr_db(x, y), expected)