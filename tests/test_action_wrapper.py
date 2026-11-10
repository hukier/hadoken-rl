"""Pruebas de StreetFighterActionWrapper."""

import numpy as np
import pytest

from hadoken_rl.envs.action_wrapper import DEFAULT_MACROS, GENESIS_BUTTONS, StreetFighterActionWrapper
from tests.fakes import FakeRetroEnv


def _pressed(vec):
    return {GENESIS_BUTTONS[i] for i in np.flatnonzero(vec)}


def _names(wrapper, name):
    return wrapper.macro_names.index(name)


def test_action_space_has_14_macros():
    w = StreetFighterActionWrapper(FakeRetroEnv())
    assert w.action_space.n == 14 == len(DEFAULT_MACROS)
    assert len(set(w.macro_names)) == 14


def test_simple_macro_holds_buttons_for_hold_frames():
    env = FakeRetroEnv()
    w = StreetFighterActionWrapper(env, hold_frames=3)
    w.reset()
    w.step(_names(w, "JUMP_FORWARD"))
    assert len(env.received) == 3
    assert all(_pressed(v) == {"UP", "RIGHT"} for v in env.received)  # personaje a la izquierda del rival
    assert env.received[0].dtype == np.int8 and env.received[0].shape == (12,)


def test_facing_flips_forward_and_back():
    env = FakeRetroEnv(agent_x=400, enemy_x=100)  # el rival está a la izquierda
    w = StreetFighterActionWrapper(env, hold_frames=1)
    w.reset()
    w.step(_names(w, "FORWARD"))
    w.step(_names(w, "BLOCK_STAND"))
    assert _pressed(env.received[0]) == {"LEFT"}
    assert _pressed(env.received[1]) == {"RIGHT"}


def test_hadoken_is_a_three_stage_sequence():
    env = FakeRetroEnv()
    w = StreetFighterActionWrapper(env)
    w.reset()
    obs, reward, term, trunc, info = w.step(_names(w, "HADOKEN"))
    stages = [_pressed(v) for v in env.received]
    assert len(stages) == 6
    assert stages[0] == {"DOWN"} and stages[2] == {"DOWN", "RIGHT"} and stages[4] == {"RIGHT", "Y"}
    assert info["macro_action"] == "HADOKEN"


def test_noop_presses_nothing():
    env = FakeRetroEnv()
    w = StreetFighterActionWrapper(env, hold_frames=2)
    w.reset()
    w.step(_names(w, "NOOP"))
    assert all(v.sum() == 0 for v in env.received)


def test_macro_stops_early_when_episode_ends():
    env = FakeRetroEnv(episode_len=2)
    w = StreetFighterActionWrapper(env)
    w.reset()
    _, _, terminated, _, _ = w.step(_names(w, "HADOKEN"))
    assert terminated and len(env.received) == 2


def test_invalid_action_and_bad_buttons():
    w = StreetFighterActionWrapper(FakeRetroEnv())
    w.reset()
    with pytest.raises(ValueError):
        w.step(14)
    with pytest.raises(ValueError):
        StreetFighterActionWrapper(FakeRetroEnv(), buttons=["A", "B"])
    with pytest.raises(ValueError):
        StreetFighterActionWrapper(FakeRetroEnv(), hold_frames=0)


def test_accepts_numpy_actions():
    w = StreetFighterActionWrapper(FakeRetroEnv())
    w.reset()
    w.step(np.int64(3))
    w.step(np.array(5))
