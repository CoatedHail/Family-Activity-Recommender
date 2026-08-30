"""
Synthetic data generator for a family-activity recommender aimed at
parents high in perfectionism / need for control.

One row = one (parent, activity) pair.
200 parents x 5 activities = 1000 rows.

ALL causal structure is declared here explicitly. See ENCODED RULES below.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

# ---------------------------------------------------------------
# ACTIVITY CATALOG
# duration in minutes; prep 0=low 1=med 2=high; location 0=home 1=out
# cost 0=free 1=low 2=med 3=high; people = min participants
# ---------------------------------------------------------------
ACTIVITIES = [
    # id, name, category, duration, prep, location, cost, people
    (1,  "Walk/run",        "Outdoor",   45,  0, 1, 0, 2),
    (2,  "Biking",          "Outdoor",   75,  0, 1, 0, 2),
    (3,  "Picnic",          "Outdoor",   120, 2, 1, 1, 3),
    (4,  "Play outside",    "Outdoor",   40,  0, 1, 0, 2),
    (5,  "Hike",            "Outdoor",   180, 1, 1, 0, 2),
    (6,  "Amusement park",  "Excursion", 480, 2, 1, 3, 3),
    (7,  "Zoo/aquarium",    "Excursion", 240, 1, 1, 2, 3),
    (8,  "Beach/pool",      "Excursion", 210, 1, 1, 1, 3),
    (9,  "Museum",          "Excursion", 150, 0, 1, 2, 2),
    (10, "Mall",            "Excursion", 150, 0, 1, 2, 2),
    (11, "Sports game",     "Excursion", 210, 1, 1, 3, 2),
    (12, "Bowling",         "Excursion", 120, 0, 1, 2, 3),
    (13, "Movie",           "Excursion", 150, 0, 1, 2, 2),
    (14, "Reading/journal", "Creative",  40,  0, 0, 0, 1),
    (15, "Arts & crafts",   "Creative",  75,  1, 0, 1, 2),
    (16, "Cooking/baking",  "Creative",  90,  2, 0, 1, 2),
    (17, "Puzzles",         "Group",     45,  0, 0, 0, 2),
    (18, "Board games",     "Group",     60,  0, 0, 0, 3),
    (19, "Karaoke",         "Group",     50,  0, 0, 0, 3),
    (20, "Tell a story",    "Group",     20,  0, 0, 0, 2),
    (21, "Call family",     "Group",     25,  0, 0, 0, 1),
]

COLS = ["activity_id", "activity_name", "category", "duration_min",
        "prep_burden", "location", "cost_tier", "people_required"]
ACT = pd.DataFrame(ACTIVITIES, columns=COLS)

INTERESTS = ["Outdoor", "Excursion", "Creative", "Group"]

N_PARENTS = 200
ACTS_PER_PARENT = 5
NOISE_SD = 8.0          # experiment knob


def complexity(row):
    """Normalized 0-1 'how many moving parts' score for an activity.

    ENCODED RULE 1: complexity is a weighted blend of duration, prep
    burden, whether it leaves the house, cost, and coordination load.
    This is the quantity that perfectionist parents react badly to.
    """
    d = min(row.duration_min / 480.0, 1.0)
    p = row.prep_burden / 2.0
    loc = row.location
    c = row.cost_tier / 3.0
    ppl = (row.people_required - 1) / 2.0
    return 0.35 * d + 0.25 * p + 0.15 * loc + 0.15 * c + 0.10 * ppl


ACT["complexity"] = ACT.apply(complexity, axis=1)


def make_parents(n):
    """ENCODED RULE 2: this is a help-seeking population, so
    perfectionism is sampled skewed high (Beta(5,2) scaled to 0-100)."""
    perf = RNG.beta(5, 2, n) * 100
    age = np.clip(RNG.normal(41, 7, n), 25, 65).round()
    family_size = RNG.choice([2, 3, 4, 5, 6], n, p=[.15, .30, .30, .18, .07])
    child_age_band = RNG.choice([0, 1, 2], n, p=[.35, .40, .25])  # 0-5, 6-11, 12-17

    # each family likes 1-3 categories
    interests = np.zeros((n, 4), dtype=int)
    for i in range(n):
        k = RNG.integers(1, 4)
        for j in RNG.choice(4, k, replace=False):
            interests[i, j] = 1

    df = pd.DataFrame({
        "parent_id": np.arange(n),
        "perfectionism": perf.round(1),
        "parent_age": age.astype(int),
        "family_size": family_size,
        "child_age_band": child_age_band,
    })
    for j, name in enumerate(INTERESTS):
        df[f"interest_{name.lower()}"] = interests[:, j]
    return df


def build(noise_sd=NOISE_SD, seed=42):
    rng = np.random.default_rng(seed)
    parents = make_parents(N_PARENTS)
    rows = []

    for _, p in parents.iterrows():
        chosen = rng.choice(len(ACT), ACTS_PER_PARENT, replace=False)
        for idx in chosen:
            a = ACT.iloc[idx]

            # context varies per attempt
            occasion = rng.choice([0, 1, 2], p=[.45, .40, .15])  # none/weekend/holiday
            time_avail = {0: rng.normal(90, 30),
                          1: rng.normal(240, 80),
                          2: rng.normal(400, 120)}[occasion]
            time_avail = float(np.clip(time_avail, 20, 600))

            pn = p.perfectionism / 100.0
            cx = a.complexity

            base = 72.0

            # RULE 3 (core mechanism): perfectionism x complexity.
            # High-perfectionism parents are penalised specifically on
            # complex activities; low-perfectionism parents barely care.
            base -= 45.0 * pn * cx

            # RULE 4: activity must fit the available time window.
            overrun = max(0.0, (a.duration_min - time_avail) / 120.0)
            base -= 14.0 * min(overrun, 2.0)

            # RULE 5: matching family interests helps, moderately.
            if p[f"interest_{a.category.lower()}"] == 1:
                base += 9.0

            # RULE 6: prep burden is harder with more people to manage.
            base -= 1.8 * a.prep_burden * (p.family_size - 2)

            # RULE 7: young children make long activities worse.
            if p.child_age_band == 0:
                base -= 8.0 * (a.duration_min / 480.0)

            # RULE 8: holidays lift the mean but add variance.
            extra_sd = 0.0
            if occasion == 2:
                base += 4.0
                extra_sd = 5.0

            outcome = base + rng.normal(0, noise_sd + extra_sd)
            outcome = float(np.clip(outcome, 0, 100))

            rows.append({
                **{k: p[k] for k in parents.columns},
                "activity_id": a.activity_id,
                "category": a.category,
                "duration_min": a.duration_min,
                "prep_burden": a.prep_burden,
                "location": a.location,
                "cost_tier": a.cost_tier,
                "people_required": a.people_required,
                "occasion": occasion,
                "time_available_min": round(time_avail),
                "outcome": round(outcome, 1),
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build()
    df.to_csv("/home/claude/activity_data.csv", index=False)
    print(df.shape)
    print(df.head())
    print("\noutcome stats:")
    print(df.outcome.describe().round(2))
