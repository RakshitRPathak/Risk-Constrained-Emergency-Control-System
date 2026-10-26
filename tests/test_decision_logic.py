"""Tests for the safety-critical decision logic: labelling rules, the safety gate and the hard constraint.

These pin behaviour at the exact thresholds (45, 6, 1, -5) where an off-by-one (`<` vs `<=`) changes a safety decision.
"""
import numpy as np
import pandas as pd
import pytest

from predict import EXAMPLE_CANDIDATE, screen, violates_hard_constraint
from simulate import LABELS, generate, label_record
from train_eval_utils import safety_gated_predict

CLASSES = np.array(["Harmful", "Inconsistent", "Ineffective", "Protective"])    # sklearn orders classes alphabetically


# ---------------------------------------------------------------- reference label rules
@pytest.mark.parametrize("dC, toward, fault, achieved, expected", [
    (-20, 0, 0, 100, "Protective"),
    (-5.01, 0, 0, 100, "Protective"),
    (-5.0, 0, 0, 100, "Ineffective"),          # a fall of exactly 5 is not "more than 5"
    (0.0, 0, 0, 100, "Ineffective"),
    (6.0, 0, 0, 100, "Ineffective"),           # a rise of exactly 6 is not "more than 6" (airflow is not toward the zone)
    (6.01, 0, 0, 100, "Harmful"),
    (1.0, 1, 0, 100, "Ineffective"),           # airflow toward the zone: rise must exceed 1
    (1.01, 1, 0, 100, "Harmful"),
    (0.5, 1, 0, 100, "Ineffective"),
    (-30, 1, 0, 100, "Protective"),            # a large fall is protective even with airflow toward the zone
    (50, 0, 1, 100, "Inconsistent"),           # sensor fault outranks Harmful: the reading cannot be trusted
    (-50, 0, 1, 100, "Inconsistent"),          # ... and outranks Protective
    (50, 0, 0, 44.99, "Inconsistent"),         # command barely executed: outcome says nothing about the action
    (50, 0, 0, 45.0, "Harmful"),               # exactly 45 is enough achievement
])
def test_label_rules_at_every_threshold(dC, toward, fault, achieved, expected):
    assert label_record(dC, toward, fault, achieved) == expected


def test_label_rule_priority_is_inconsistent_then_harmful_then_protective():
    # Harmful beats Protective only through the dC thresholds, which cannot both hold; check the ordering where inputs overlap.
    assert label_record(10, 1, 1, 10) == "Inconsistent"
    assert label_record(10, 1, 0, 100) == "Harmful"


# ---------------------------------------------------------------- safety gate
def _proba(**kw):
    p = np.array([[kw.get(c, 0.0) for c in CLASSES]], dtype=float)
    return p


def test_protective_is_returned_only_at_or_above_tau():
    tau = 0.7
    assert safety_gated_predict(_proba(Protective=0.70, Ineffective=0.30), CLASSES, tau)[0] == "Protective"
    assert safety_gated_predict(_proba(Protective=0.699, Ineffective=0.301), CLASSES, tau)[0] == "Ineffective"


def test_a_protective_argmax_below_tau_falls_back_to_the_best_non_protective_class():
    # Protective is the single most likely class (0.45) but not confident enough: it must NOT be admitted.
    out = safety_gated_predict(_proba(Protective=0.45, Harmful=0.30, Ineffective=0.25), CLASSES, 0.8)
    assert out[0] == "Harmful"


def test_fallback_never_returns_protective_for_any_tau_below_one_shot_random_inputs():
    rng = np.random.default_rng(0)
    proba = rng.dirichlet(np.ones(4), size=2000)
    pi = int(np.where(CLASSES == "Protective")[0][0])
    for tau in (0.0, 0.3, 0.5, 0.9, 1.0, 1.5):
        out = safety_gated_predict(proba, CLASSES, tau)
        assert ((out == "Protective") == (proba[:, pi] >= tau)).all()      # Protective exactly when confident enough
        assert set(out) <= set(CLASSES)


def test_tau_above_one_admits_nothing_and_tau_zero_admits_everything():
    proba = np.random.default_rng(1).dirichlet(np.ones(4), size=500)
    assert "Protective" not in set(safety_gated_predict(proba, CLASSES, 1.0001))
    assert set(safety_gated_predict(proba, CLASSES, 0.0)) == {"Protective"}


def test_gate_does_not_modify_its_input_and_handles_any_class_order():
    proba = _proba(Protective=0.9, Harmful=0.1)
    before = proba.copy()
    safety_gated_predict(proba, CLASSES, 0.5)
    assert np.array_equal(proba, before)                                    # the internal masking works on a copy
    shuffled = np.array(["Protective", "Inconsistent", "Harmful", "Ineffective"])
    p = np.array([[0.6, 0.1, 0.2, 0.1]])
    assert safety_gated_predict(p, shuffled, 0.5)[0] == "Protective"
    assert safety_gated_predict(np.array([[0.3, 0.1, 0.5, 0.1]]), shuffled, 0.5)[0] == "Harmful"


def test_fallback_is_never_protective_even_when_every_other_class_has_zero_probability():
    # Protective listed FIRST and probability exactly 1: with tau above 1 the gate must still refuse, and the fallback
    # must pick some other class (a mask of 0 instead of -1 would tie at 0 and hand the argmax back to Protective).
    first = np.array(["Protective", "Harmful", "Ineffective", "Inconsistent"])
    out = safety_gated_predict(np.array([[1.0, 0.0, 0.0, 0.0]]), first, 1.0001)
    assert out[0] != "Protective"


# ---------------------------------------------------------------- hard constraint
class _ExplodingModel:
    classes_ = CLASSES

    def predict_proba(self, X):
        raise AssertionError("the model must not be consulted when a hard constraint already forbids the action")


class _AlwaysProtective:
    classes_ = CLASSES

    def predict_proba(self, X):
        return np.tile(_proba(Protective=1.0), (len(X), 1))


def _candidate(**over):
    return pd.DataFrame([{**EXAMPLE_CANDIDATE, **over}])


@pytest.mark.parametrize("occupancy, toward, forbidden", [(1, 1, True), (1, 0, False), (0, 1, False), (0, 0, False)])
def test_hard_constraint_truth_table(occupancy, toward, forbidden):
    assert violates_hard_constraint(_candidate(occupancy=occupancy, airflow_direction_to_protected=toward)) is forbidden


def test_hard_constraint_rejects_without_asking_the_model():
    r = screen(_candidate(occupancy=1, airflow_direction_to_protected=1), _ExplodingModel(), tau=0.0)
    assert r["rejected_by_hard_constraint"] and r["decision"] == "Harmful" and r["probabilities"] is None


def test_hard_constraint_overrides_a_model_that_is_certain_the_action_is_protective():
    r = screen(_candidate(occupancy=1, airflow_direction_to_protected=1), _AlwaysProtective(), tau=0.0)
    assert r["decision"] != "Protective"
    ok = screen(_candidate(occupancy=0, airflow_direction_to_protected=1), _AlwaysProtective(), tau=0.5)
    assert ok["decision"] == "Protective"                                   # without the constraint the same model is admitted


# ---------------------------------------------------------------- simulator
def test_generator_is_deterministic_and_seeded():
    a, b, c = generate(300, 7), generate(300, 7), generate(300, 8)
    pd.testing.assert_frame_equal(a, b)
    assert not a.drop(columns="record_id").equals(c.drop(columns="record_id"))


def test_generated_data_obeys_the_dataset_invariants():
    df = generate(2000, 42)
    assert len(df) == 2000 and df.isna().sum().sum() == 0
    assert (df.protected_zone != df.source_zone).all()
    assert (df.hop_distance == (df.source_zone - df.protected_zone).abs()).all() and df.hop_distance.between(1, 3).all()
    assert df.actuator_trust.between(0, 1).all() and df.command_achievement_pct.between(0, 100).all()
    assert df.communication_delay_ms.between(20, 900).all()
    assert set(df.outcome) == set(LABELS)                                   # all four classes occur
    relabelled = [label_record(d, a, s, c) for d, a, s, c in
                  zip(df.dC_protected, df.airflow_direction_to_protected, df.sensor_fault, df.command_achievement_pct)]
    assert list(df.outcome) == relabelled                                   # stored labels are exactly the stated rules


def test_a_sensor_fault_always_yields_inconsistent():
    df = generate(3000, 3)
    assert (df.loc[df.sensor_fault == 1, "outcome"] == "Inconsistent").all()


# ---------------------------------------------------------------- shipped model
def test_shipped_model_gives_a_valid_decision_and_its_gate_is_stricter_than_argmax():
    import pickle
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent / "outputs" / "selected_model.pkl"
    try:
        bundle = pickle.loads(path.read_bytes())
        model, tau = bundle["pipeline"], bundle["protective_tau"]
        model.predict_proba(_candidate())
    except Exception as e:                                                  # wrong scikit-learn version for the pickle
        pytest.skip(f"shipped model not loadable with this scikit-learn: {e}")
    r = screen(_candidate(), model, tau)
    assert r["decision"] in LABELS
    assert 0.0 < tau <= 1.0
    X = generate(1500, 99).drop(columns=["record_id", "dC_protected", "command_achievement_pct", "outcome"])
    proba = model.predict_proba(X)
    gated = safety_gated_predict(proba, model.classes_, tau)
    p_protective = proba[:, list(model.classes_).index("Protective")]
    assert ((gated == "Protective") == (p_protective >= tau)).all()         # admitted exactly when confident enough
