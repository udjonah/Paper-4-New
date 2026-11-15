"""
04_weighted_robustness.py
Robustness check: re-estimates the RO1 and RO2 models with survey weights.

Three weighting schemes are compared:
  1. Unweighted                    : the main estimates
  2. Survey-weighted               : WBES strict-eligibility weights as published.
                                     They sum to each country's firm population, so
                                     the largest economies (India) dominate.
  3. Survey-weighted, equal country: weights rescaled to sum to 1 within each
                                     country, so every country counts equally while
                                     firms stay representative within their country.

Each is run for: main fixed effects (country x sector), alternative fixed effects
(country x region), all seven countries and excluding India.

Requires: pandas, numpy, matplotlib, statsmodels, linearmodels
Input: output/wbes_epay_collateral_clean.csv  (from 01_build_clean_data.py)
Output: output/robustness/weighted_estimates.csv and fig7_weighted_robustness.png
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from linearmodels.iv import IV2SLS

warnings.filterwarnings("ignore")
OUT = Path("output/robustness")
OUT.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ data
df = pd.read_csv("output/wbes_epay_collateral_clean.csv")
df["epay"] = df["epay_received_pct"] / 100
df["z"] = df["iv_loo_epay_pct"] / 100
df["c_sector"] = df["country"] + "_" + df["sector_code"].astype(str)
df["c_region"] = df["country"] + "_" + df["region_code"].astype(str)
CTRL = ["ln_employees", "ln_firm_age", "woman_top_manager", "foreign_owned",
        "exporter", "audited", "website"]
FE = {"Country×sector FE": ["c_sector"], "Country×region FE": ["c_region"]}
OUTCOMES = {"Incidence (RO1)": ("collateral_required", "in_rq1_sample"),
            "Collateral/loan, log (RO2)": ("ln_collateral_to_loan", "in_rq2_sample")}
WEIGHTS = ["Unweighted", "Survey-weighted", "Survey-weighted, equal country"]


def design(d, fe):
    """Controls plus fixed-effect dummies, dropping linearly dependent columns."""
    X = pd.concat([d[CTRL]] + [pd.get_dummies(d[f], prefix=f, drop_first=True) for f in fe],
                  axis=1).astype(float)
    X = X.loc[:, X.std() > 0]
    X.insert(0, "const", 1.0)
    keep, M = [], np.empty((len(X), 0))
    for c in X.columns:
        trial = np.column_stack([M, X[c].values])
        if np.linalg.matrix_rank(trial) > M.shape[1]:
            M, keep = trial, keep + [c]
    return X[keep]


def make_weights(d, scheme):
    if scheme == "Unweighted":
        return None
    w = d["weight_strict"].astype(float)
    if scheme == "Survey-weighted, equal country":
        w = w / w.groupby(d["country"]).transform("sum")
    return w / w.mean()  # normalise to mean 1 (does not change the estimates)


def fit(d, y, fe, scheme):
    d = d.dropna(subset=[y, "epay", "z", "weight_strict"] + CTRL)
    X, w = design(d, fe), make_weights(d, scheme)
    cl = pd.factorize(d["iv_cell"])[0]
    r = IV2SLS(d[y], X, d["epay"], d["z"], weights=w).fit(cov_type="clustered", clusters=cl)
    # Weighted first stage, cluster-robust F on the single excluded instrument
    fs = sm.WLS(d["epay"], X.assign(z=d["z"]), weights=1 if w is None else w).fit(
        cov_type="cluster", cov_kwds={"groups": cl})
    return dict(coef=r.params["epay"], se=r.std_errors["epay"], n=len(d),
                first_stage_F=(fs.params["z"] / fs.bse["z"]) ** 2)


rows = []
for outcome, (y, flag) in OUTCOMES.items():
    base = df[df[flag] == 1]
    for sample, d in [("All seven countries", base), ("Excluding India", base[base.country != "India"])]:
        for fe_name, fe in FE.items():
            for scheme in WEIGHTS:
                res = fit(d, y, fe, scheme)
                rows.append(dict(outcome=outcome, sample=sample, fixed_effects=fe_name,
                                 weighting=scheme, **res))
est = pd.DataFrame(rows)
est["ci_low"] = est.coef - 1.96 * est.se
est["ci_high"] = est.coef + 1.96 * est.se
est["p_value"] = 2 * (1 - __import__("scipy").stats.norm.cdf(np.abs(est.coef / est.se)))
est.to_csv(OUT / "weighted_estimates.csv", index=False)
print(est.round(3).to_string(index=False))

# ------------------------------------------------------------------ figure
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COLORS = {"Unweighted": "#2a78d6", "Survey-weighted": "#eb6834",
          "Survey-weighted, equal country": "#1baf7a"}   # categorical slots 1-3
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "figure.facecolor": SURF,
    "axes.facecolor": SURF, "savefig.facecolor": SURF, "savefig.dpi": 200})

specs = [(s, f) for s in ["All seven countries", "Excluding India"] for f in FE]
fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
offsets = dict(zip(WEIGHTS, [-0.22, 0, 0.22]))
for ax, outcome in zip(axes, OUTCOMES):
    g = est[est.outcome == outcome]
    strong = g[g.first_stage_F >= 10]
    lo, hi = strong.ci_low.min(), strong.ci_high.max()
    pad = 0.08 * (hi - lo)
    ax.set_xlim(min(lo, 0) - pad, max(hi, 0) + pad)
    for i, (sample, fe_name) in enumerate(specs):
        for scheme in WEIGHTS:
            r = g[(g["sample"] == sample) & (g.fixed_effects == fe_name) & (g.weighting == scheme)].iloc[0]
            yy = i + offsets[scheme]
            if r.first_stage_F >= 10:
                ax.errorbar(r.coef, yy, xerr=1.96 * r.se, fmt="o", color=COLORS[scheme],
                            ecolor=COLORS[scheme], elinewidth=2, ms=7, capsize=0,
                            mec=SURF, mew=1.5)
            else:
                ax.text(0.01, yy, f"  {scheme}: weak instrument (F = {r.first_stage_F:.1f})",
                        transform=ax.get_yaxis_transform(), va="center", fontsize=8,
                        color=INK2, style="italic")
    for k in range(1, len(specs)):
        ax.axhline(k - 0.5, color=GRID, lw=1)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(range(len(specs)), [f"{s}\n{f}" for s, f in specs])
    ax.set_ylim(len(specs) - 0.5, -0.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("2SLS effect of 0% → 100% e-payments")
    ax.set_title(outcome, loc="left", fontsize=12, fontweight="bold", color=INK, pad=22)
    ax.text(0, 1.02, "Point estimate and 95% CI by weighting scheme", transform=ax.transAxes,
            fontsize=9, color=INK2, va="bottom")
from matplotlib.lines import Line2D
handles = [Line2D([0], [0], marker="o", color=COLORS[w], lw=2, ms=7, mec=SURF, mew=1.5) for w in WEIGHTS]
fig.legend(handles, WEIGHTS, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.04))
fig.text(0.01, 0.005, "Survey weights: WBES strict-eligibility weights. Equal country: weights rescaled to sum to 1 "
         "within each country. SEs clustered by instrument cell. Estimates with first-stage F < 10 are not drawn.",
         fontsize=8.5, color=INK2)
fig.tight_layout(rect=(0, 0.1, 1, 1))
fig.savefig(OUT / "fig7_weighted_robustness.png")
print("saved", OUT / "fig7_weighted_robustness.png")
