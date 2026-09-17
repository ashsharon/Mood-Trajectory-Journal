"""
pipeline.py
-----------
Top-level orchestration: journal entries -> feature time series -> trajectory
analysis -> plain-language narrative summary.
"""

import os
import json
from trajectory import build_feature_timeseries, analyze_trajectory
from narrative import NarrativeGenerator


class MoodTrajectoryPipeline:
    def __init__(self, api_key: str | None = None):
        self.narrative_generator = NarrativeGenerator(api_key=api_key)

    def run(self, entries: list[dict], recent_window: int = 4) -> dict:
        timeseries = build_feature_timeseries(entries)
        analysis = analyze_trajectory(timeseries, recent_window=recent_window)
        narrative = self.narrative_generator.generate(analysis)

        return {
            "timeseries": timeseries,
            "analysis": analysis,
            "narrative": narrative["summary"],
            "suggest_support": narrative["suggest_support"]
        }


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "data", "sample_journal_entries.json")) as f:
        entries = json.load(f)

    pipeline = MoodTrajectoryPipeline()
    result = pipeline.run(entries)

    print("NARRATIVE:", result["narrative"])
    print("SUGGEST SUPPORT:", result["suggest_support"])
    print("\nTRENDS:")
    print(json.dumps(result["analysis"]["trends"], indent=2))
