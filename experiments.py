"""Experiments for the parent activity recommender."""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, cross_val_score

from generate import build, ACT

CAT_MAP = {"Outdoor": 0, "Excursion": 1, "Creative": 2, "Group": 3}


def prep(df, drop=()):
    d = df.copy()
    d["category"] = d["category"].map(CAT_MAP)
    y = d.pop("outcome").values
    groups = d.pop("parent_id").values
    d = d.drop(columns=[c for c in drop if c in d.columns])
    return d.values, y, groups, list(d.columns)


MODELS = {
    "Baseline (mean)":   DummyRegressor(strategy="mean"),
    "Linear regression": LinearRegression(),
    "Ridge":             Ridge(alpha=1.0),
    "Decision tree":     DecisionTreeRegressor(max_depth=6, random_state=0),
    "k-NN (k=10)":       make_pipeline(StandardScaler(), KNeighborsRegressor(10)),
    "SVR (RBF)":         make_pipeline(StandardScaler(), SVR()),
    "Random forest":     RandomForestRegressor(n_estimators=300, random_state=0),
    "Gradient boosting": GradientBoostingRegressor(random_state=0),
}


def evaluate(X, y, groups, model, k=5):
    cv = GroupKFold(n_splits=k)
    r2 = cross_val_score(model, X, y, cv=cv, groups=groups, scoring="r2")
    mae = -cross_val_score(model, X, y, cv=cv, groups=groups,
                           scoring="neg_mean_absolute_error")
    return r2.mean(), r2.std(), mae.mean()


def exp1_model_comparison(df):
    X, y, g, _ = prep(df)
    print("\n=== EXPERIMENT 1: model comparison (5-fold GroupKFold by parent) ===")
    print(f"{'Model':<20}{'R2 mean':>10}{'R2 sd':>9}{'MAE':>9}")
    print("-" * 48)
    for name, m in MODELS.items():
        r2, sd, mae = evaluate(X, y, g, m)
        print(f"{name:<20}{r2:>10.3f}{sd:>9.3f}{mae:>9.2f}")


def exp2_ablation(df):
    print("\n=== EXPERIMENT 2: feature ablation (Random forest) ===")
    sets = {
        "All features":          (),
        "No perfectionism":      ("perfectionism",),
        "Demographics only":     ("perfectionism", "duration_min", "prep_burden",
                                  "location", "cost_tier", "people_required",
                                  "category", "activity_id"),
        "Activity only":         ("perfectionism", "parent_age", "family_size",
                                  "child_age_band", "interest_outdoor",
                                  "interest_excursion", "interest_creative",
                                  "interest_group"),
    }
    print(f"{'Feature set':<22}{'R2 mean':>10}{'R2 sd':>9}")
    print("-" * 41)
    for name, drop in sets.items():
        X, y, g, _ = prep(df, drop=drop)
        r2, sd, _ = evaluate(X, y, g,
                             RandomForestRegressor(n_estimators=300, random_state=0))
        print(f"{name:<22}{r2:>10.3f}{sd:>9.3f}")


def exp3_noise_sweep():
    print("\n=== EXPERIMENT 3: robustness to self-report noise ===")
    print(f"{'Noise SD':<12}{'RF R2':>9}{'Linear R2':>12}")
    print("-" * 33)
    for sd_noise in [2, 5, 8, 12, 18, 25]:
        d = build(noise_sd=sd_noise, seed=42)
        X, y, g, _ = prep(d)
        rf, _, _ = evaluate(X, y, g,
                            RandomForestRegressor(n_estimators=200, random_state=0))
        lin, _, _ = evaluate(X, y, g, LinearRegression())
        print(f"{sd_noise:<12}{rf:>9.3f}{lin:>12.3f}")


def exp4_ranking(df):
    """Does the model rank a parent's options better than a global
    popularity ranking? This is the question the product actually asks."""
    from scipy.stats import spearmanr
    X, y, g, cols = prep(df)
    cv = GroupKFold(n_splits=5)
    model_rho, pop_rho = [], []

    global_mean = df.groupby("activity_id").outcome.mean()

    for tr, te in cv.split(X, y, groups=g):
        m = RandomForestRegressor(n_estimators=300, random_state=0).fit(X[tr], y[tr])
        pred = m.predict(X[te])
        te_df = df.iloc[te].copy()
        te_df["pred"] = pred
        te_df["pop"] = te_df.activity_id.map(global_mean)
        for _, grp in te_df.groupby("parent_id"):
            if len(grp) < 3:
                continue
            model_rho.append(spearmanr(grp.pred, grp.outcome).statistic)
            pop_rho.append(spearmanr(grp["pop"], grp.outcome).statistic)

    print("\n=== EXPERIMENT 4: per-parent ranking quality (Spearman rho) ===")
    print(f"Personalized model : {np.nanmean(model_rho):.3f}")
    print(f"Popularity baseline: {np.nanmean(pop_rho):.3f}")


if __name__ == "__main__":
    df = build()
    exp1_model_comparison(df)
    exp2_ablation(df)
    exp3_noise_sweep()
    exp4_ranking(df)
