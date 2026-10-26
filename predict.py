"""
predict.py - Screen one candidate micro-intervention with the saved model.
Usage: python predict.py

`screen()` holds the decision logic so it can be tested without loading a model from disk.
"""
import pickle

import pandas as pd

from train_eval_utils import safety_gated_predict

TREAT = {"Protective": "ADMIT to bounded micro-intervention test (escalate only if measured response confirms)",
         "Harmful": "REJECT before execution",
         "Ineffective": "HOLD as alternative / diagnostic action",
         "Inconsistent": "REVIEW: extra sensing, fault attribution or operator approval"}

HARD_CONSTRAINT_MESSAGE = "REJECT: airflow toward an occupied protected zone (hard constraint)"

EXAMPLE_CANDIDATE = {
    "source_zone": 1, "protected_zone": 3, "hop_distance": 2,
    "initial_hazard_concentration": 32.0, "spatial_gradient": -0.6, "temporal_gradient": -0.2,
    "airflow_speed_mps": 1.4, "airflow_direction_to_protected": 0, "pressure_diff_pa": 3.0,
    "oxygen_pct": 20.1, "door_open": 0, "blocked_path": 0, "sensor_fault": 0, "actuator_fault": 0,
    "communication_delay_ms": 80.0, "actuator_type": "damper", "intervention_magnitude_pct": 25.0,
    "intervention_duration_s": 8.0, "actuator_trust": 0.86, "occupancy": 1, "evacuation_route_exposure": 0,
}


def violates_hard_constraint(candidate: pd.DataFrame) -> bool:
    """The protected-zone rule that no model output may override: an occupied zone with airflow toward it."""
    return bool(candidate.occupancy.iloc[0] == 1 and candidate.airflow_direction_to_protected.iloc[0] == 1)


def screen(candidate: pd.DataFrame, model, tau: float) -> dict:
    """Returns {"decision", "rejected_by_hard_constraint", "probabilities"}.

    The hard constraint is checked BEFORE the model is consulted, so a model bug or a pickle from the wrong
    version can never admit a candidate that the rule forbids.
    """
    if violates_hard_constraint(candidate):
        return {"decision": "Harmful", "rejected_by_hard_constraint": True, "probabilities": None}
    proba = model.predict_proba(candidate)
    label = safety_gated_predict(proba, model.classes_, tau)[0]
    return {"decision": str(label), "rejected_by_hard_constraint": False,
            "probabilities": {c: round(float(p), 3) for c, p in zip(model.classes_, proba[0])}}


def main(path="outputs/selected_model.pkl"):
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    result = screen(pd.DataFrame([EXAMPLE_CANDIDATE]), bundle["pipeline"], bundle["protective_tau"])
    if result["rejected_by_hard_constraint"]:
        print(HARD_CONSTRAINT_MESSAGE)
    else:
        print("Class probabilities:", result["probabilities"])
        print(f"Decision: {result['decision']} -> {TREAT[result['decision']]}")


if __name__ == "__main__":
    main()
