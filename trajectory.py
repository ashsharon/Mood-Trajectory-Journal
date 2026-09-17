"""
trajectory.py
-------------
Aggregates per-entry linguistic features into a time series and detects trend
shifts using simple, transparent statistics (not a black-box classifier):

- Rolling average to smooth noise between individual entries
- Linear regression slope over a recent window vs. the full history, to detect
  directional trends (not just single bad days)
- A simple flag when recent sentiment/negation/absolutist trends move
  significantly beyond the person's own baseline (self-referential, not
  compared to population norms)

This is intentionally simple and explainable rather than a deep model, since the
goal is a transparent trajectory the person (or a report reader) can inspect -
not an opaque risk score.
"""

import numpy as np
from features import extract_features


TREND_FEATURES = ["sentiment_compound", "first_person_singular_ratio",
                   "negation_ratio", "social_word_ratio", "word_count"]


def build_feature_timeseries(entries: list[dict]) -> list[dict]:
    """entries: list of {date, text}, sorted chronologically. Returns entries with features attached."""
    result = []
    for entry in sorted(entries, key=lambda e: e["date"]):
        feats = extract_features(entry["text"])
        result.append({"date": entry["date"], "text": entry["text"], **feats})
    return result


def _linear_trend_slope(values: list[float]) -> float:
    """Simple least-squares slope of values against their index (time order)."""
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values))
    y = np.array(values)
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def analyze_trajectory(timeseries: list[dict], recent_window: int = 4) -> dict:
    """
    Computes overall + recent trend slopes for key features, and flags a
    'shift' if the recent window differs meaningfully from the person's own
    earlier baseline. All thresholds are simple z-score style comparisons
    against the person's OWN history - never compared to other people.
    """
    if len(timeseries) < 3:
        return {
            "sufficient_data": False,
            "note": "Need at least 3 entries to compute a trajectory.",
            "trends": {}
        }

    trends = {}
    for feature in TREND_FEATURES:
        values = [entry[feature] for entry in timeseries]

        baseline = values[:-recent_window] if len(values) > recent_window else values[:1]
        recent = values[-recent_window:]

        baseline_mean = float(np.mean(baseline))
        baseline_std = float(np.std(baseline)) or 1e-6
        recent_mean = float(np.mean(recent))

        z_shift = (recent_mean - baseline_mean) / baseline_std
        overall_slope = _linear_trend_slope(values)

        trends[feature] = {
            "baseline_mean": round(baseline_mean, 4),
            "recent_mean": round(recent_mean, 4),
            "z_shift_from_baseline": round(z_shift, 3),
            "overall_slope": round(overall_slope, 5)
        }

    # A conservative, transparent flag: sentiment down AND first-person-singular up
    # AND social words down, all meaningfully shifted from the person's own baseline.
    concerning_shift = (
        trends["sentiment_compound"]["z_shift_from_baseline"] < -1.0 and
        trends["first_person_singular_ratio"]["z_shift_from_baseline"] > 1.0 and
        trends["social_word_ratio"]["z_shift_from_baseline"] < -0.5
    )

    return {
        "sufficient_data": True,
        "num_entries": len(timeseries),
        "trends": trends,
        "concerning_shift_detected": concerning_shift
    }


if __name__ == "__main__":
    import json, os

    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "data", "sample_journal_entries.json")) as f:
        entries = json.load(f)

    ts = build_feature_timeseries(entries)
    analysis = analyze_trajectory(ts)
    print(json.dumps(analysis, indent=2))
