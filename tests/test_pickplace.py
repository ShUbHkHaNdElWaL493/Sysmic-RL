import numpy as np
import pytest

from stub_robot import write_stub_robot
from sysmic_rl.imitation.dataset import load_demos
from sysmic_rl.imitation.collect import collect
from sysmic_rl.pickplace import PickPlaceConfig, SysmicPickPlaceEnv


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    c = PickPlaceConfig()
    c.scene.robot_xml = write_stub_robot(tmp_path_factory.mktemp("robot"))
    return c


def test_spaces_and_determinism(cfg):
    e1, e2 = SysmicPickPlaceEnv(cfg), SysmicPickPlaceEnv(cfg)
    o1, _ = e1.reset(seed=3)
    o2, _ = e2.reset(seed=3)
    assert o1.shape == e1.observation_space.shape and np.allclose(o1, o2)
    a = e1.action_space.sample()
    s1, s2 = e1.step(a), e2.step(a)
    assert np.allclose(s1[0], s2[0]) and np.isfinite(s1[1])


def test_support_surface_detected(cfg):
    e = SysmicPickPlaceEnv(cfg)
    e.reset(seed=0)
    lo, hi, z = e.surface.bbox(e.data)
    assert e.surface.body_name == "lift_carriage_link" and hi[0] - lo[0] > 0.2 and e.surface.area > 0.05
    # object starts ON the surface
    assert abs(e.object_pos()[2] - z) < 0.01 and lo[0] < e.object_pos()[0] < hi[0]


def test_table_spawned_in_ranges_and_clear_of_robot(cfg):
    e = SysmicPickPlaceEnv(cfg)
    sc = cfg.scene
    for seed in range(5):
        e.reset(seed=seed)
        assert sc.table_z[0] <= e.place_z <= sc.table_z[1]
        assert not e._table_collides(clearance=0.0)   # no real penetration (arms sag slightly while settling)
        assert abs(e.goal_pos[2] - e.place_z) < 1e-9
        assert np.all(np.abs(e.goal_pos[:2] - e.table_center) <= np.array(sc.table_half_xy))


def test_bimanual_for_plate_and_bowl_only(cfg):
    e = SysmicPickPlaceEnv(cfg)
    for name, both in [("plate", True), ("bowl", True), ("glass", False), ("mug", False)]:
        _, info = e.reset(seed=2, options={"object": name})
        assert (info["arm"] == "both") == both and (len(e.sides) == 2) == both
    e.reset(seed=2, options={"object": "plate", "arm": "left"})
    assert e.sides == ("left",)


def test_bimanual_plan_has_opposite_sides(cfg):
    e = SysmicPickPlaceEnv(cfg)
    e.reset(seed=3, options={"object": "plate"})
    plan, o = e.grasp_plan(), e.object_pos()
    assert plan["left"][0][1] > o[1] > plan["right"][0][1]
    assert set(e.action_mask[1:13]) == {1.0}


def test_expert_runs_bimanual_and_single(cfg):
    from sysmic_rl.imitation.expert import ScriptedExpert
    e, ex = SysmicPickPlaceEnv(cfg), ScriptedExpert()
    for name in ("plate", "glass"):
        e.reset(seed=1, options={"object": name}); ex.reset()
        for _ in range(60):
            a = ex.act(e)
            assert a.shape == (13,) and np.all(np.abs(a) <= 1.0)
            e.step(a)


def test_inactive_arm_is_frozen(cfg):
    e = SysmicPickPlaceEnv(cfg)
    e.reset(seed=1, options={"arm": "left", "object": "glass"})
    t0 = e.target[e.sl["right"]].copy()
    for _ in range(20):
        e.step(np.ones(13, dtype=np.float32))
    assert np.allclose(e.target[e.sl["right"]], t0)
    assert not np.allclose(e.target[e.sl["left"]], t0)


@pytest.mark.parametrize("name", ["plate", "glass", "bowl", "mug"])
def test_objects_settle_on_source_surface(cfg, name):
    e = SysmicPickPlaceEnv(cfg)
    e.reset(seed=0, options={"object": name})
    assert abs(e.object_pos()[2] - e.src_z) < 0.01
    assert e.object_tilt() < 0.1


def test_collect_and_load(cfg, tmp_path):
    out = tmp_path / "d.h5"
    collect(2, str(out), cfg, noise=0.0, seed=0, keep_failures=True, log=lambda *_: None)
    d, n = load_demos(str(out), successful_only=False)
    assert d["obs"].shape[1] == SysmicPickPlaceEnv(cfg).observation_space.shape[0]
    assert d["actions"].shape[1] == 13 and len(d["obs"]) == len(d["rewards"])


def test_reachability_and_seed(cfg):
    e = SysmicPickPlaceEnv(cfg)
    e.reset(seed=0, options={"object": "glass"})
    plan = e.grasp_plan()
    assert e.reachable(plan, source=True)
    lift, q5, err = e.seed_config({s: (plan[s][0], plan[s][1]) for s in e.sides})
    assert err < 0.2 and len(q5[e.side]) == 5
    if e.surface_moves:   # carriage-mounted source: the lift must not change during the seed move
        assert abs(lift - e.joint_pos()[0]) < 1e-9


def test_probe_runs(cfg, capsys):
    import sys
    from sysmic_rl.pickplace import probe
    sys.argv = ["probe", "--robot-xml", cfg.scene.robot_xml, "--samples", "400"]
    probe.main()
    out = capsys.readouterr().out
    assert "source surface" in out and "reachability" in out
