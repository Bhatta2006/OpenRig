import numpy as np
from openrig.core.solver import Solver, SolveSettings
from openrig.core.rotations import solve_rotations, swing
from rigs import make

PINS = {"foot.L", "foot.R", "toe.L", "toe.R", "toe_end.L", "toe_end.R"}


def setup():
    bones, m, sk, P0 = make()
    return sk, P0, Solver(sk, P0, SolveSettings())


def test_no_target_is_identity():
    sk, P0, sol = setup()
    P = sol.solve({}, PINS)
    assert np.allclose(P, P0, atol=1e-4)


def test_lengths_preserved_and_feet_planted():
    sk, P0, sol = setup()
    ix = sk.index
    tgt = {"hand.L": P0[ix["hand.L"]] + np.array([-0.3, -0.5, 0.4])}
    P = sol.solve(tgt, PINS)
    assert sol.max_length_error(P) < 1e-2
    for p in PINS:
        assert np.allclose(P[ix[p]], P0[ix[p]], atol=1e-9)


def test_reachable_hand_target_is_hit():
    sk, P0, sol = setup()
    ix = sk.index
    t = P0[ix["hand.L"]] + np.array([-0.2, -0.3, 0.2])
    P = sol.solve({"hand.L": t}, PINS)
    assert np.linalg.norm(P[ix["hand.L"]] - t) < 0.02


def test_far_hand_pull_moves_the_body():
    sk, P0, sol = setup()
    ix = sk.index
    t = P0[ix["hand.L"]] + np.array([0.0, -0.9, 0.0])  # out of arm's reach
    P = sol.solve({"hand.L": t}, PINS)
    moved = np.linalg.norm(P[ix["head"]] - P0[ix["head"]])
    assert moved > 0.05, "torso should lean toward an out-of-reach hand"
    assert sol.max_length_error(P) < 1e-2


def test_hips_drag_down_bends_knees_forward():
    sk, P0, sol = setup()
    ix = sk.index
    t = P0[ix["hips"]] + np.array([0, 0, -0.25])
    P = sol.solve({"hips": t}, PINS)
    assert abs(P[ix["hips"]][2] - t[2]) < 0.03
    fwd = np.array([0, -1.0, 0])
    for s in "LR":
        hip, knee, ankle = P[ix[f"thigh.{s}"]], P[ix[f"shin.{s}"]], P[ix[f"foot.{s}"]]
        assert (knee - (hip + ankle) / 2) @ fwd > 0.02, f"knee {s} must point forward"
    assert sol.max_length_error(P) < 1e-2


def test_knee_never_hyperextends_when_pulled_back():
    sk, P0, sol = setup()
    ix = sk.index
    # push the hips up past leg reach; leg must straighten, not invert
    t = P0[ix["hips"]] + np.array([0, 0, 0.2])
    P = sol.solve({"hips": t}, PINS)
    fwd = np.array([0, -1.0, 0])
    for s in "LR":
        hip, knee, ankle = P[ix[f"thigh.{s}"]], P[ix[f"shin.{s}"]], P[ix[f"foot.{s}"]]
        assert (knee - (hip + ankle) / 2) @ fwd > -1e-3


def test_elbow_bends_backward_not_forward():
    sk, P0, sol = setup()
    ix = sk.index
    # sweep the hand across the front of the chest -> elbow must point back/outward (+Y), not through the body
    t = np.array([0.05, -0.25, 1.4])
    P = sol.solve({"hand.L": t}, PINS)
    sh, el, wr = P[ix["upperarm.L"]], P[ix["forearm.L"]], P[ix["hand.L"]]
    assert (el - (sh + wr) / 2) @ np.array([0, 1.0, 0]) > 0.01


def test_symmetry():
    sk, P0, sol = setup()
    ix = sk.index
    tl = P0[ix["hand.L"]] + np.array([-0.1, -0.4, 0.3])
    tr = P0[ix["hand.R"]] + np.array([0.1, -0.4, 0.3])
    PL = sol.solve({"hand.L": tl}, PINS)
    PR = sol.solve({"hand.R": tr}, PINS)
    mirror = np.array([-1, 1, 1.0])
    for a in sk.names:
        if a.endswith(".L"):
            b = a[:-2] + ".R"
            assert np.allclose(PL[ix[a]] * mirror, PR[ix[b]], atol=2e-3), a


def test_floor_clamps_free_foot():
    bones, m, sk, P0 = make()
    sol = Solver(sk, P0, SolveSettings(floor=True, floor_height=0.0))
    ix = sk.index
    P = sol.solve({"hips": P0[ix["hips"]] + [0, 0, -0.3]}, {"foot.R", "toe.R", "toe_end.R"})
    for k in ("foot.L", "toe.L", "toe_end.L"):
        assert P[ix[k]][2] >= -2e-3


def test_rotations_are_proper_and_aim_at_solved_joints():
    sk, P0, sol = setup()
    ix = sk.index
    P = sol.solve({"hand.L": P0[ix["hand.L"]] + [-0.2, -0.5, 0.3]}, PINS)
    R_old = {e.key: np.eye(3) for e in sk.edges}
    R_new = solve_rotations(sk, P0, P, R_old)
    for e in sk.edges:
        assert np.allclose(R_new[e.key].T @ R_new[e.key], np.eye(3), atol=1e-8)
        assert np.linalg.det(R_new[e.key]) > 0.999
    # single-child limb bones point exactly at their solved joint
    for key in ("forearm.L", "shin.L", "shin.R", "upperarm.L"):
        e = sk.edge_by_key[key]
        got = R_new[key] @ (P0[e.b] - P0[e.a])
        want = P[e.b] - P[e.a]
        cos = got @ want / (np.linalg.norm(got) * np.linalg.norm(want))
        assert cos > 0.999, key


def test_swing_edge_cases():
    a = np.array([0, 0, 1.0])
    assert np.allclose(swing(a, a), np.eye(3))
    R = swing(a, -a)
    assert np.allclose(R @ a, -a)
    b = np.array([1.0, 2, 3]) / np.linalg.norm([1, 2, 3])
    assert np.allclose(swing(a, b) @ a, b)


def test_pinned_joints_are_never_moved_by_hinges():
    """Lock in Place pins everything but one leg; the other knee and both elbows must not move."""
    sk, P0, sol = setup()
    ix = sk.index
    free = {"shin.L", "foot.L", "toe.L", "toe_end.L"}
    pins = set(sk.names) - free
    t = {k: P0[ix[k]] + np.array([0.0, -0.05, 0.1]) for k in ("foot.L", "toe.L", "toe_end.L")}
    P = sol.solve(t, pins)
    for k in pins:
        assert np.allclose(P[ix[k]], P0[ix[k]], atol=1e-9), k


# ---------------- pose prior plumbing (no trained weights needed) ----------------
from openrig.core import prior as PR   # noqa: E402


class _Identity:
    """A 'perfect' prior: returns what it is given. Reconstruction must then be a no-op."""
    def project(self, x):
        return np.asarray(x)


def _bent_pose():
    sk, P0, sol = setup()
    ix = sk.index
    t = P0[ix["hand.L"]] + np.array([-0.2, -0.45, 0.25])
    P = sol.solve({"hand.L": t, "hips": P0[ix["hips"]] + [0, 0, -0.12]}, PINS)
    return sk, P0, P


def test_prior_goal_reconstruction_is_consistent():
    sk, P0, P = _bent_pose()
    g = PR.PriorGoal(sk, P0, _Identity())
    G = g.goals(P)
    # joints on the canonical chains must be reproduced (to the extent bone lengths still match P0's)
    for name in ("head", "head_top", "forearm.L", "hand.L", "shin.R", "foot.R"):
        assert np.linalg.norm(G[sk.index[name]] - P[sk.index[name]]) < 5e-3, name


def _mate(name):
    return name[:-1] + ("R" if name.endswith("L") else "L") if name.endswith((".L", ".R")) else name


def test_feature_mirror_matches_geometric_mirror():
    sk, P0, P = _bent_pose()
    c = PR.Canon(sk)
    sign = PR.forward_sign(c, P0)
    f = PR.features(c, P, sign)
    Pm = np.zeros_like(P)
    for i, n in enumerate(sk.names):
        Pm[sk.index[_mate(n)]] = P[i] * np.array([-1, 1, 1.0])   # reflect across the sagittal plane
    assert np.allclose(PR.mirror_features(f), PR.features(c, Pm, sign), atol=1e-6)
    assert np.allclose(PR.mirror_features(PR.mirror_features(f)), f)


def test_solver_with_prior_keeps_all_hard_constraints():
    sk, P0, sol0 = setup()
    ix = sk.index
    s = SolveSettings(prior=_Identity(), prior_strength=1.0)
    sol = Solver(sk, P0, s)
    t = P0[ix["hand.L"]] + np.array([-0.2, -0.4, 0.3])
    P = sol.solve({"hand.L": t}, PINS)
    assert sol.max_length_error(P) < 1e-2
    for p in PINS:
        assert np.allclose(P[ix[p]], P0[ix[p]], atol=1e-9)
