"""
narrative.py
------------
Generates a plain-language summary of the detected trajectory patterns using an
LLM. This module is deliberately constrained:

- It NEVER diagnoses a condition or names a clinical label.
- It describes patterns in the person's OWN language use over time, in the
  person's own words where possible (quoting sparingly), not inferred emotions.
- If a concerning shift is detected, it gently encourages the person to consider
  talking to a trusted person or professional - it does not tell them what's
  "wrong" with them.
- If the trend looks positive or neutral, it says so plainly instead of hedging.

This mirrors how a supportive, non-clinical tool should communicate: describing
patterns, not diagnosing people.
"""

import json
import anthropic


NARRATIVE_SYSTEM_PROMPT = """You are summarizing linguistic trends from someone's private \
journal entries over time, based on statistical analysis already performed on their language \
(sentiment, self-focus, social references, etc). You are NOT a therapist or doctor.

Strict rules:
- NEVER diagnose or name a clinical condition (e.g. do not say "depression", "anxiety \
disorder", etc).
- NEVER claim certainty about how the person feels - describe PATTERNS IN LANGUAGE, not \
inferred emotional states as fact. Use phrasing like "your recent entries show..." not \
"you are feeling...".
- If the data shows a concerning shift (declining sentiment, increased self-focus, reduced \
social references), gently and non-alarmingly suggest that if this reflects how they've \
genuinely been feeling, it could be worth talking to someone they trust or a professional -\
 phrase this as an option, not a directive.
- If the data shows stable or positive trends, say so plainly and warmly, no false alarm.
- Keep the tone warm, human, and non-clinical - like a thoughtful friend summarizing \
patterns, not a report.
- 3-5 sentences maximum.

Respond ONLY with valid JSON, no markdown fences, in this exact shape:
{
  "summary": "the plain-language summary, following all rules above",
  "suggest_support": true | false
}
"""


class NarrativeGenerator:
    def __init__(self, api_key: str | None = None, model: str = "claude-sonnet-4-6"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate(self, trajectory_analysis: dict) -> dict:
        if not trajectory_analysis.get("sufficient_data"):
            return {
                "summary": trajectory_analysis.get("note", "Not enough entries yet to show a trend."),
                "suggest_support": False
            }

        user_message = (
            "Here is the statistical trend analysis of someone's journal entries over time:\n\n"
            f"{json.dumps(trajectory_analysis, indent=2)}"
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=400,
            system=NARRATIVE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}]
        )

        raw = response.content[0].text.strip().replace("```json", "").replace("```", "")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"summary": raw, "suggest_support": trajectory_analysis.get("concerning_shift_detected", False)}


if __name__ == "__main__":
    import os
    from trajectory import build_feature_timeseries, analyze_trajectory

    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "data", "sample_journal_entries.json")) as f:
        entries = json.load(f)

    ts = build_feature_timeseries(entries)
    analysis = analyze_trajectory(ts)

    generator = NarrativeGenerator()
    result = generator.generate(analysis)
    print(json.dumps(result, indent=2))
