import numpy as np
from active.simulation_utils import make_normals, orient_normals
from driver.legacy.models import Driver
from hypothesis import given, settings
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats


def safe_normalize(x: np.ndarray) -> np.ndarray:
    """ Tries to normalize the input, but returns the input if anything goes wrong. """
    norm = np.linalg.norm(x)
    if np.isfinite(norm) and norm != 0.0:
        x /= norm
    return x


@settings(deadline=None)
@given(
    actions=arrays(
        dtype=np.float32,
        shape=(1, 2, 50, 2),
        elements=floats(min_value=-1, max_value=1, allow_nan=False, width=32),
    ),
    reward=arrays(
        dtype=np.float32,
        shape=(4),
        elements=floats(min_value=-1, max_value=1, allow_nan=False, width=32),
    ),
)
def test_orient_normals(actions: np.ndarray, reward: np.ndarray):
    reward = safe_normalize(reward)

    _, normals = make_normals(inputs=actions, sim=Driver(), use_equiv=False)
    value_diffs = reward @ normals.T
    prefs = value_diffs > 0

    oriented_normals = orient_normals(normals, preferences=prefs)

    assert np.all(reward @ oriented_normals.T == np.abs(value_diffs))
