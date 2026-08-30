"""Does personalization pay off? Sweep the strength of the
perfectionism x complexity interaction."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
import generate as G
from experiments import prep

def build_with(beta, seed=42):
    """Rebuild data with a custom interaction coefficient."""
    rng = np.random.default_rng(seed)
    parents = G.make_parents(G.N_PARENTS); rows=[]
    for _, p in parents.iterrows():
        for idx in rng.choice(len(G.ACT), G.ACTS_PER_PARENT, replace=False):
            a = G.ACT.iloc[idx]
            occ = rng.choice([0,1,2], p=[.45,.40,.15])
            ta = float(np.clip({0:rng.normal(90,30),1:rng.normal(240,80),
                                2:rng.normal(400,120)}[occ],20,600))
            pn, cx = p.perfectionism/100.0, a.complexity
            base = 72.0 - beta*pn*cx
            base -= 14.0*min(max(0.0,(a.duration_min-ta)/120.0),2.0)
            if p[f"interest_{a.category.lower()}"]==1: base += 9.0
            base -= 1.8*a.prep_burden*(p.family_size-2)
            if p.child_age_band==0: base -= 8.0*(a.duration_min/480.0)
            esd = 5.0 if occ==2 else 0.0
            if occ==2: base += 4.0
            rows.append({**{k:p[k] for k in parents.columns},
                "activity_id":a.activity_id,"category":a.category,
                "duration_min":a.duration_min,"prep_burden":a.prep_burden,
                "location":a.location,"cost_tier":a.cost_tier,
                "people_required":a.people_required,"occasion":occ,
                "time_available_min":round(ta),
                "outcome":float(np.clip(base+rng.normal(0,8.0+esd),0,100))})
    return pd.DataFrame(rows)

print(f"{'beta':<8}{'model rho':>11}{'popularity rho':>16}{'gain':>8}")
print("-"*43)
for beta in [0,15,30,45,60,80,100]:
    df = build_with(beta); X,y,g,_ = prep(df)
    mr, pr = [], []
    gm = df.groupby("activity_id").outcome.mean()
    for tr,te in GroupKFold(n_splits=5).split(X,y,groups=g):
        m = RandomForestRegressor(n_estimators=200,random_state=0).fit(X[tr],y[tr])
        t = df.iloc[te].copy(); t["pred"]=m.predict(X[te]); t["pop"]=t.activity_id.map(gm)
        for _,grp in t.groupby("parent_id"):
            if len(grp)<3: continue
            mr.append(spearmanr(grp.pred,grp.outcome).statistic)
            pr.append(spearmanr(grp["pop"],grp.outcome).statistic)
    a,b = np.nanmean(mr), np.nanmean(pr)
    print(f"{beta:<8}{a:>11.3f}{b:>16.3f}{a-b:>8.3f}")
