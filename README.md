# Mood Trajectory Journal: Longitudinal Linguistic Mood Tracking

A journaling tool that analyzes *trajectories* in language over time, rather than
classifying single entries as positive/negative. It extracts linguistic features
studied in psycholinguistic research (self-focus, negation, social word use, lexical
diversity, sentiment) from each entry, tracks how they shift over time relative to
the person's own baseline, and generates a plain-language, non-diagnostic summary.

**This is the unique feature relative to typical sentiment-analysis journaling apps:**
time-series modeling of language, not snapshot classification — and the design is
built around not overclaiming what the model can tell you about a person.

## ⚠️ Important: read before using or presenting this

This project touches mental health, so a few things matter for both responsible use
and your methodology/ethics section:

- **This is not a diagnostic tool** and should never be presented as one. It surfaces
  *linguistic patterns*, not psychological states. The narrative generator is
  explicitly instructed to never name a clinical condition.
- **All comparisons are self-referential.** The system never compares one person's
  language to population norms or to other users — only to that person's own earlier
  baseline. This avoids pathologizing normal individual variation.
- **The sample data is synthetic**, written for this project to demonstrate a
  declining trajectory for demo purposes. Do not present it as real clinical data.
- If you deploy this for real personal use, journal entries are **only stored in
  memory** in the current app (nothing persists to disk) — call this out explicitly
  if you extend it to add persistence, since journal content is sensitive data.
- The app includes a built-in prompt toward professional/trusted support when a
  concerning shift is detected — this is a deliberate design choice worth discussing
  in an ethics section (duty of care in wellbeing-adjacent tools).

## Architecture

```
Journal entries (date + text)
        │
        ▼
features.py    ──► per-entry linguistic features (sentiment, self-focus, negation,
        │            social words, lexical diversity, absolutist language)
        ▼
trajectory.py   ──► builds time series, computes trend slopes + z-shift from the
        │             person's own baseline (transparent stats, not a black box)
        ▼
narrative.py     ──► LLM generates a plain-language, non-diagnostic summary,
        │              constrained to never name a clinical condition
        ▼
pipeline.py        ──► orchestrates the above
        │
        ▼
app.py               ──► Streamlit UI with trend charts + supportive messaging
```

## Project structure

```
mood-trajectory-journal/
├── app.py                          # Streamlit frontend
├── pipeline.py                       # Orchestration
├── features.py                        # Per-entry linguistic feature extraction
├── trajectory.py                       # Time-series trend + baseline-shift detection
├── narrative.py                          # LLM narrative summary (non-diagnostic)
├── requirements.txt
├── .env.example
└── data/
    └── sample_journal_entries.json       # 10 synthetic entries, declining trajectory
```

## Setup (VS Code / local)

1. **Open this folder in VS Code.**

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate      # macOS/Linux
   venv\Scripts\activate         # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set your Anthropic API key:**
   ```bash
   export ANTHROPIC_API_KEY=your-key-here     # macOS/Linux
   set ANTHROPIC_API_KEY=your-key-here        # Windows (cmd)
   ```

5. **Run the app:**
   ```bash
   streamlit run app.py
   ```
   Opens at `http://localhost:8501` pre-loaded with 10 sample entries showing a
   declining trajectory. Click "Analyze Trajectory" to see the trend charts and
   narrative summary. Add your own entries via the expander at the top.

6. **Or run from the command line:**
   ```bash
   python pipeline.py
   ```

## Testing individual components

```bash
python features.py       # test feature extraction on 3 sample entries
python trajectory.py       # test trend/baseline-shift analysis
python narrative.py          # test full pipeline: trend analysis + LLM narrative
```

## Extending it (ideas for your report / future work section)

- Replace the lexicon-based feature set with LIWC (Linguistic Inquiry and Word Count)
  categories if you have a license - it's the standard tool in this research area and
  would strengthen your methodology section considerably.
- Validate the feature set against a labeled longitudinal dataset (several exist in
  academic mental-health NLP research, usually requiring a data use agreement).
- Add local persistence (encrypted at rest) if building toward real personal use,
  with an explicit data retention and deletion policy.
- Compare the transparent statistical trend detection against a trained time-series
  model (e.g. an LSTM over feature sequences) and discuss the accuracy/interpretability
  trade-off - a good discussion point for a masters-level report.
- Add inter-rater validation: have the LLM's narrative summaries reviewed by people
  with clinical training for tone and appropriateness, and report agreement.

## Notes

- First run downloads the NLTK VADER sentiment lexicon (~1MB, cached after).
- The "concerning shift" flag in `trajectory.py` is intentionally conservative
  (requires shifts across three independent features simultaneously) to avoid
  false alarms from a single unusual entry.
