"""The bundled learned pose prior, checked on poses from motion-capture subjects it never saw."""
import pathlib

import numpy as np
import pytest

from openrig.core import prior as PR
from openrig.core.prior import PosePrior

DATA = pathlib.Path(__file__).parent / "data" / "heldout_features.npz"


@pytest.fixture(scope="module")
def prior():
    p = PosePrior.load()
    assert p is not None, "openrig/data/pose_prior.npz is missing"
    return p


@pytest.fixture(scope="module")
def poses():
    return np.load(DATA)["X"].astype(float)


def noisy(X, sigma, seed=1):
    rng = np.random.default_rng(seed)
    Y = X + rng.normal(size=X.shape) * sigma / np.sqrt(3)
    Y = Y.reshape(len(X), -1, 3)
    return (Y / np.linalg.norm(Y, axis=-1, keepdims=True)).reshape(X.shape)


def test_shapes_and_unit_directions(prior, poses):
    out = prior.project(poses)
    assert out.shape == poses.shape == (1500, 3 * PR.N_DIRS)
    norms = np.linalg.norm(out.reshape(len(out), -1, 3), axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-5)
    single = prior.project(poses[0])
    assert np.allclose(single, out[0], atol=1e-6)


def test_removes_noise_on_unseen_subjects(prior, poses):
    # Below ~0.15 the network's own reconstruction error (~0.002) is as big as the noise, so it
    # only helps once a pose is meaningfully off, which is when the solver needs it.
    for sigma, need in ((0.2, 0.15), (0.3, 0.25)):
        y = noisy(poses, sigma)
        before = ((y - poses) ** 2).mean()
        after = ((prior.project(y) - poses) ** 2).mean()
        assert after < (1 - need) * before, (sigma, before, after)


def test_leaves_real_poses_alone(prior, poses):
    err = ((prior.project(poses) - poses) ** 2).mean()
    assert err < 0.004


def test_prefers_real_to_impossible(prior, poses):
    """A random arrangement of directions is pulled toward a real pose more than a real one is."""
    rng = np.random.default_rng(3)
    junk = rng.normal(size=poses.shape).reshape(len(poses), -1, 3)
    junk = (junk / np.linalg.norm(junk, axis=-1, keepdims=True)).reshape(poses.shape)
    moved_junk = np.linalg.norm(prior.project(junk) - junk, axis=-1).mean()
    moved_real = np.linalg.norm(prior.project(poses) - poses, axis=-1).mean()
    assert moved_junk > 5 * moved_real


def test_mirror_augmentation_keeps_poses_plausible(prior, poses):
    m = PR.mirror_features(poses)
    assert ((prior.project(m) - m) ** 2).mean() < 0.006


def test_bundled_weights_are_small():
    size = pathlib.Path(PR.WEIGHTS).stat().st_size
    assert size < 400_000
