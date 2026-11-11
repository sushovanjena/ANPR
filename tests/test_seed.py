import numpy as np

from anpr.utils.seed import seed_everything


def test_seed_is_reproducible():
    seed_everything(0)
    first = np.random.rand(3)
    seed_everything(0)
    assert np.allclose(first, np.random.rand(3))
