import numpy as np

from experiments.run_relative_noise import relative_noise_sd


def test_relative_noise_sd_scales_with_the_data():
    generator = np.random.default_rng(19)
    data = generator.normal(size=(80, 12))

    original = relative_noise_sd(data, eta=0.2, neighbor_order=10)
    rescaled = relative_noise_sd(7.0 * data, eta=0.2, neighbor_order=10)

    assert np.isclose(rescaled, 7.0 * original)


def test_relative_noise_sd_is_zero_at_zero_level():
    data = np.arange(120, dtype=float).reshape(20, 6)

    assert relative_noise_sd(data, eta=0.0, neighbor_order=5) == 0.0


def test_relative_noise_uses_the_requested_nonself_neighbor_order():
    data = np.arange(12, dtype=float)[:, None]

    sigma = relative_noise_sd(data, eta=1.0, neighbor_order=2)

    assert np.isclose(sigma, 1.0 / np.sqrt(2.0))
