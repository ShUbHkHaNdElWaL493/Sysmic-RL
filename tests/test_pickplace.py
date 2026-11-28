import warnings

import numpy as np
import pytest

from sysmic_rl.imitation.collect import collect
from sysmic_rl.imitation.dataset import load_demos
from sysmic_rl.pickplace import PickPlaceConfig, SysmicPickPlaceEnv

warnings.filterwarnings("ignore", message="No reachable table pose")


@pytest.fixture(scope="module")
def cfg():
    c = PickPlaceConfig()          # the real mjcf/sysmic_v2_scene.xml
    c.scene.reach_samples = 5000   # keep env construction fast
    return c


@pytest.fixture(scope="module")
def env(cfg):
    return SysmicPickPlaceEnv(cfg)


def test_spaces_and_determinism(cfg):
    e1, e2 = SysmicPickPlaceEnv(cfg), SysmicPickPlaceEnv(cfg)
    o1, _ = e1.reset(seed=3)
    o2, _ = e2.reset(seed=3)
    assert o1.shape == e1.observation_space.shape and np.allclose(o1, o2)
    assert e1.obs_layout["others_pos"].stop - e1.obs_layout["others_pos"].start == 3 * (len(e1.object_names) - 1)
    a = e1.action_space.sample()
    s1, s2 = e1.step(a), e2.step(a)
    assert np.allclose(s1[0], s2[0]) and np.isfinite(s1[1])


@pytest.mark.parametrize("seed", range(5))
def test_all_utensils_settle_on_shelf(env, seed):
    env.reset(seed=seed)
    cx, cy, z, hx, hy = env.shelf
    spans = []
    for n in env.object_names:
        b = env.obj_body[n]
        p = env.data.xpos[b]
        assert abs(p[2] - z) < 0.01, n                      # resting on the shelf (not fallen, not ejected)
        assert abs(p[0] - cx) <= hx and abs(p[1] - cy) <= hy, n
        assert env.data.xmat[b][8] > np.cos(0.1), n          # upright
        w = env.specs[n].footprint_y
        spans.append((p[1] - w, p[1] + w))
    spans.sort()
    assert all(a[1] <= b[0] + 1e-3 for a, b in zip(spans, spans[1:]))   # no overlap along the shelf


def test_utensil_order_is_randomised(env):
    orders = set()
    for seed in range(8):
        env.reset(seed=seed)
        orders.add(tuple(sorted(env.object_names, key=lambda n: env.data.xpos[env.obj_body[n]][1])))
    assert len(orders) > 1


def test_table_pose_in_ranges_and_clear_of_robot(env, cfg):
    sc = cfg.scene
    for seed in range(10):
        _, info = env.reset(seed=seed)
        if info["reach_fallback"]:
            continue   # fallback keeps the last candidate, which may be anywhere
        assert sc.table_z[0] <= env.place_z <= sc.table_z[1]
        assert sc.table_yaw[0] <= env.table_yaw <= sc.table_yaw[1]
        assert not env._table_collides(clearance=0.0)
        assert abs(env.goal_pos[2] - env.place_z) < 1e-9
        assert np.all(np.abs(env.world_to_table(env.goal_pos[:2])) <= env.table_half)


def test_table_options_and_frames(env):
    env.reset(seed=0, options={"table_pos": (0.45, 0.05, 0.8), "table_yaw": 0.7})
    assert np.allclose(env.table_center, (0.45, 0.05)) and env.place_z == 0.8 and env.table_yaw == 0.7
    uv = np.array([0.1, -0.05])
    assert np.allclose(env.world_to_table(env.table_to_world(uv)), uv)
    # the table's top face really is where the env thinks it is
    top = env.data.geom_xpos[env.table_geom][2] + env.model.geom_size[env.table_geom][2]
    assert abs(top - 0.8) < 1e-9


def test_table_collision_check_sees_the_base(env):
    # a table pushed into the robot's base must be detected (base_link and table are both world-welded,
    # so MuJoCo would not report a contact between them)
    env.reset(seed=0)
    env._place_table((0.15, 0.0), 0.6)
    assert env._table_collides(clearance=0.0)
    env._place_table((1.5, 0.0), 0.6)
    assert not env._table_collides()


def test_bimanual_for_plate_and_bowl_only(env):
    for name, both in [("plate", True), ("bowl", True), ("glass", False), ("mug", False)]:
        _, info = env.reset(seed=2, options={"object": name})
        assert (info["arm"] == "both") == both and (len(env.sides) == 2) == both
    env.reset(seed=2, options={"object": "plate", "arm": "left"})
    assert env.sides == ("left",)


def test_bimanual_plan_has_opposite_sides(env):
    env.reset(seed=3, options={"object": "plate"})
    plan, o = env.grasp_plan(), env.object_pos()
    assert plan["left"][0][1] > o[1] > plan["right"][0][1]
    assert set(env.action_mask[1:13]) == {1.0}


def test_touching_inactive_utensil_is_bad_contact(env):
    env.reset(seed=0, options={"object": "plate"})
    d, m = env.data, env.model
    finger = int(np.where(env.geom_finger["left"])[0][0])
    other = int(np.where(env.geom_obj == env.object_names.index("glass"))[0][0])
    active = int(np.where(env.geom_obj == env.active_idx)[0][0])
    for g in (other, active):
        d.ncon = 1
        d.contact.geom[0] = (finger, g)
        d.contact.dist[0] = -0.001
        n_bad = len(env._bad_contacts(ignore_baseline=False))
        assert n_bad == (1 if g == other else 0)
    env.reset(seed=0)   # leave the shared env in a clean state


def test_expert_runs_bimanual_and_single(env):
    from sysmic_rl.imitation.expert import ScriptedExpert
    ex = ScriptedExpert()
    for name in ("plate", "glass"):
        env.reset(seed=1, options={"object": name}); ex.reset()
        for _ in range(60):
            a = ex.act(env)
            assert a.shape == (13,) and np.all(np.abs(a) <= 1.0)
            env.step(a)


def test_inactive_arm_is_frozen(env):
    env.reset(seed=1, options={"arm": "left", "object": "glass"})
    t0 = env.target[env.sl["right"]].copy()
    for _ in range(20):
        env.step(np.ones(13, dtype=np.float32))
    assert np.allclose(env.target[env.sl["right"]], t0)
    assert not np.allclose(env.target[env.sl["left"]], t0)


def test_collect_and_load(cfg, tmp_path):
    out = tmp_path / "d.h5"
    collect(2, str(out), cfg, noise=0.0, seed=0, keep_failures=True, log=lambda *_: None)
    d, n = load_demos(str(out), successful_only=False)
    assert d["obs"].shape[1] == SysmicPickPlaceEnv(cfg).observation_space.shape[0]
    assert d["actions"].shape[1] == 13 and len(d["obs"]) == len(d["rewards"])


def test_seed_config(env):
    env.reset(seed=0, options={"object": "plate"})
    plan = env.grasp_plan()
    lift, q5, err = env.seed_config({s: (plan[s][0], plan[s][1]) for s in env.sides})
    assert np.isfinite(err) and env.lo[0] <= lift <= env.hi[0]
    assert set(q5) == set(env.sides) and all(len(q) == 5 for q in q5.values())


def test_probe_runs(cfg, capsys):
    import sys
    from sysmic_rl.pickplace import probe
    sys.argv = ["probe", "--scene-xml", cfg.scene.scene_xml, "--samples", "400"]
    probe.main()
    out = capsys.readouterr().out
    assert "shelf" in out and "reachability" in out
