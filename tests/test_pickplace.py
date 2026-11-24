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


def test_inactive_arm_is_frozen(cfg):
    e = SysmicPickPlaceEnv(cfg)
    e.reset(seed=1, options={"arm": "left", "object": "glass"})
    t0 = e.target[e.sl["right"]].copy()
    for _ in range(20):
        e.step(np.ones(13, dtype=np.float32))
    assert np.allclose(e.target[e.sl["right"]], t0)
    assert not np.allclose(e.target[e.sl["left"]], t0)


@pytest.mark.parametrize("name", ["plate", "glass", "bowl", "mug"])
def test_objects_settle_on_table(cfg, name):
    e = SysmicPickPlaceEnv(cfg)
    e.reset(seed=0, options={"object": name})
    assert abs(e.object_pos()[2] - cfg.scene.table_top_z) < 0.01
    assert e.object_tilt() < 0.1


def test_collect_and_load(cfg, tmp_path):
    out = tmp_path / "d.h5"
    collect(2, str(out), cfg, noise=0.0, seed=0, keep_failures=True, log=lambda *_: None)
    d, n = load_demos(str(out), successful_only=False)
    assert d["obs"].shape[1] == SysmicPickPlaceEnv(cfg).observation_space.shape[0]
    assert d["actions"].shape[1] == 13 and len(d["obs"]) == len(d["rewards"])
