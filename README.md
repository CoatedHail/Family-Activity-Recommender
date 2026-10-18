# Family Activity Recommender — Simulation Code

Code for the paper **"A Personalized Family-Activity Recommendation System for Reducing Planning Burden in Parents with High Perfectionistic Concerns and Intolerance of Uncertainty"** (accepted to NLAIM 2026).

The system recommends a family activity and builds a delegation step into the recommendation, so the parent carrying the planning load hands off execution instead of managing it. This repository holds the simulation study used to evaluate the recommender.

> **All data here is synthetic.** The generating rules are written out explicitly in `generate.py` (Section 3.3 of the paper). The results describe how well each model recovers that structure, not how real parents behave.

## Results

1,000 simulated household–activity pairs (200 households × 5 activities), evaluated with 5-fold cross-validation grouped by household so that no household appears in both train and test.

| Experiment | Finding |
|---|---|
| Model comparison | Random forest R² = 0.720, gradient boosting 0.713, linear regression 0.682, against a mean baseline at ≈ 0 |
| Feature ablation | Activity features carry most of the signal (0.584 on their own); demographics alone score below the baseline (−0.053) |
| Noise robustness | Random forest leads at low self-report noise; linear regression overtakes it once noise SD passes ~15 points |
| Ranking quality | Personalized ranking (Spearman ρ = 0.716) is barely better than a global popularity ranking (0.698) |

The last result is structural, not a matter of model choice. When a trait is represented as one severity score that scales how demanding an activity feels, it preserves every household's ranking order. A trait has to *reorder* preferences, not just rescale them, before personalization beats popularity. `sweep.py` shows how the gap changes as that interaction gets stronger.

## Files

| File | What it does |
|---|---|
| `generate.py` | Activity catalog and synthetic data generator (all causal rules declared here) |
| `experiments.py` | Experiments 1–4: model comparison, feature ablation, noise sweep, ranking evaluation |
| `sweep.py` | Interaction-strength sweep (Table 5) |

## Running

Requires Python 3.10+.

```bash
pip install -r requirements.txt
python experiments.py   # Experiments 1–4
python sweep.py         # Table 5
```

The random seed is fixed (42), so the output reproduces the tables in the paper exactly.
