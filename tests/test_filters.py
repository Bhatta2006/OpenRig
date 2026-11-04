import numpy as np
from openrig.core import filters as F


def test_smooth_reduces_noise_and_keeps_trend():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 200)
    clean = np.sin(x)
    noisy = clean + rng.normal(0, 0.1, x.size)
    r = F.kernel_radius(3)
    out = F.smooth_valid(np.pad(noisy, r, mode="edge"), 3)
    assert out.shape == noisy.shape
    assert np.abs(out - clean)[10:-10].mean() < np.abs(noisy - clean)[10:-10].mean()


def test_smooth_constant_is_constant():
    out = F.smooth_valid(np.full(50, 2.5), 2.0)
    assert np.allclose(out, 2.5)


def _walk(n=60):
    t = np.arange(n)
    x = np.zeros(n)
    z = np.zeros(n)
    z[20:40] = 0.0      # planted
    z[:20] = np.linspace(0.2, 0.0, 20)
    z[40:] = np.linspace(0.0, 0.2, n - 40)
    return np.stack([x, np.zeros(n), z], 1)


def test_planted_foot_has_no_slide():
    pos = _walk()
    assert F.sliding_segments(pos, (0, 0, 1), 0.02, 0.02) == []


def test_sliding_foot_detected():
    pos = _walk()
    pos[20:40, 0] = np.linspace(0, 0.15, 20)  # 15 cm skate while on the ground
    segs = F.sliding_segments(pos, (0, 0, 1), 0.02, 0.02)
    assert len(segs) == 1 and 18 <= segs[0]["start"] <= 22 and segs[0]["drift"] > 0.05


def test_rotation_outlier_found():
    n = 40
    ang = np.linspace(0, 1.0, n)
    Q = np.stack([np.cos(ang / 2), np.sin(ang / 2), 0 * ang, 0 * ang], 1)
    assert F.rotation_outliers(Q) == []
    pop = 0.6
    Q[17] = [np.cos((ang[17] + pop) / 2), np.sin((ang[17] + pop) / 2), 0, 0]
    assert 17 in F.rotation_outliers(Q)


def test_quaternion_sign_flip_is_not_a_pop():
    n = 30
    ang = np.linspace(0, 1.0, n)
    Q = np.stack([np.cos(ang / 2), np.sin(ang / 2), 0 * ang, 0 * ang], 1)
    Q[10:15] *= -1
    assert F.rotation_outliers(Q) == []


def test_warp_time_is_monotonic_and_local():
    t = np.linspace(0, 40, 801)
    for shift in (-2.8, -1.0, 0.7, 2.8):
        w = F.warp_time(t, 20, shift, 4)
        assert np.all(np.diff(w) > 0), shift
        assert abs(w[400] - 20 - shift) < 1e-9
        assert np.array_equal(w[:300], t[:300])                  # beyond the radius: exactly untouched
    assert F.max_retime_shift(4) == 2.8


def test_falloff_reaches_zero_at_radius():
    assert F.falloff(0, 5) == 1.0
    assert F.falloff(5, 5) == 0.0 and F.falloff(-7, 5) == 0.0
    assert 0 < F.falloff(3, 5) < F.falloff(1, 5) < 1.0
    assert np.all(F.falloff(np.arange(-6, 7), 5) >= 0)
