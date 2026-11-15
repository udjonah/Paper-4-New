"""
02_visualise_objectives.py
Figures for the three research objectives, built from the clean dataset
produced by 01_build_clean_data.py.

RO1  Incidence   : effect of e-payment share on whether collateral is required
RO2  Intensity / composition : effect on collateral-to-loan ratio and collateral type
RO3  Heterogeneity: whether effects differ by firm age, manager gender,
                    sector and country collateral intensity

Requires: pandas, numpy, matplotlib, statsmodels, linearmodels
Run: python 02_visualise_objectives.py   (figures saved to output/figures)
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
from linearmodels.iv import IV2SLS

warnings.filterwarnings("ignore")
DATA = Path("output/wbes_epay_collateral_clean.csv")
FIG = Path("output/figures")
FIG.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- style
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"           # categorical slots 1 and 2
TERCILE = ["#86b6ef", "#2a78d6", "#104281"]   # ordinal blue ramp (low -> high)
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "figure.facecolor": SURF,
    "axes.facecolor": SURF, "savefig.facecolor": SURF, "savefig.dpi": 200,
})


def title(ax, text, sub):
    ax.set_title(text, loc="left", fontsize=12, fontweight="bold", color=INK, pad=22)
    ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=9, color=INK2, va="bottom")


def wmean(x, w):
    m = x.notna() & w.notna()
    return np.average(x[m], weights=w[m]) if m.any() else np.nan


# ---------------------------------------------------------------- data
df = pd.read_csv(DATA)
df["epay"] = df["epay_received_pct"] / 100
df["z"] = df["iv_loo_epay_pct"] / 100
df["c_sector"] = df["country"] + "_" + df["sector_code"].astype(str)
df["c_region"] = df["country"] + "_" + df["region_code"].astype(str)
CTRL = "ln_employees + ln_firm_age + woman_top_manager + foreign_owned + exporter + audited + website"
# Main specification: country x sector fixed effects. The instrument varies at the
# country x sector x region level, so identification comes from regional variation
# within the same country-sector. Adding country x region FE as well absorbs most of
# that variation (first-stage F falls to about 6), so that version is a robustness
# check that needs weak-instrument-robust inference, not the main model.
FE_MAIN = ["c_sector"]
FE_ALT = ["c_region"]
rq1 = df[df.in_rq1_sample == 1].copy()
rq2 = df[df.in_rq2_sample == 1].copy()


CTRL_VARS = CTRL.split(" + ")


def design(d, fe=None):
    """Controls + country x sector and country x region dummies, with any
    linearly dependent columns removed (the two sets of dummies overlap at
    the country level, so some columns are redundant by construction)."""
    fe = FE_MAIN if fe is None else fe
    X = pd.concat([d[CTRL_VARS]] + [pd.get_dummies(d[f], prefix=f, drop_first=True) for f in fe],
                  axis=1).astype(float)
    X = X.loc[:, X.std() > 0]
    X.insert(0, "const", 1.0)
    keep, M = [], np.empty((len(X), 0))
    for c in X.columns:                      # greedy rank check, column by column
        trial = np.column_stack([M, X[c].values])
        if np.linalg.matrix_rank(trial) > M.shape[1]:
            M, keep = trial, keep + [c]
    return X[keep]


def iv(data, y, fe=None):  # returns coef, se, n, first-stage F
    """2SLS: y on e-payment share, instrumented by leave-one-out cell mean."""
    d = data.dropna(subset=[y] + CTRL_VARS)
    r = IV2SLS(d[y], design(d, fe), d.epay, d.z).fit(cov_type="clustered", clusters=pd.factorize(d.iv_cell)[0])
    return r.params["epay"], r.std_errors["epay"], len(d), r.first_stage.diagnostics["f.stat"].iloc[0]


def ols(data, y):
    d = data.dropna(subset=[y] + CTRL_VARS)
    X = design(d).assign(epay=d.epay)
    r = sm.OLS(d[y], X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.iv_cell)[0]})
    return r.params["epay"], r.bse["epay"], len(d)


# ================================================================ Figure 1
# Context: collateral incidence, size and e-payment use by country (survey-weighted)
ctx = []
for c, g in df.groupby("country"):
    b = g[g.collateral_required.notna()]
    r = g.collateral_to_loan_pct.dropna()
    ctx.append(dict(country=c,
                    incidence=100 * wmean(b.collateral_required, b.weight_strict),
                    ratio_med=r.median(), ratio_q1=r.quantile(.25), ratio_q3=r.quantile(.75),
                    epay=wmean(g.epay_received_pct, g.weight_strict)))
ctx = pd.DataFrame(ctx).sort_values("incidence")
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
y = np.arange(len(ctx))
axes[0].barh(y, ctx.incidence, color=BLUE, height=0.55)
axes[0].set_yticks(y, ctx.country); axes[0].set_xlim(0, 100); axes[0].set_xlabel("% of borrowers")
title(axes[0], "Collateral required", "Most recent loan, survey-weighted")
axes[1].errorbar(ctx.ratio_med, y, xerr=[ctx.ratio_med - ctx.ratio_q1, ctx.ratio_q3 - ctx.ratio_med],
                 fmt="o", color=BLUE, ecolor=BLUE, elinewidth=2, capsize=0, ms=7)
axes[1].axvline(100, color=INK2, lw=1, ls="--"); axes[1].set_xlabel("Collateral value as % of loan")
title(axes[1], "Size of the pledge", "Median and interquartile range")
axes[2].barh(y, ctx.epay, color=BLUE, height=0.55); axes[2].set_xlim(0, 100)
axes[2].set_xlabel("% of sales"); title(axes[2], "Sales received electronically", "All firms, survey-weighted")
for ax in axes: ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(FIG / "fig1_context_by_country.png"); plt.close(fig)

# ================================================================ Figure 2 (RO1)
# Raw pattern: share of borrowers asked for collateral, by e-payment band
bands = [0, 20, 40, 60, 80, 100.01]
labels = ["0–19", "20–39", "40–59", "60–79", "80–100"]
fig, ax = plt.subplots(figsize=(7.5, 4.5))
for name, d, col in [("All seven countries", rq1, BLUE), ("Excluding India", rq1[rq1.country != "India"], ORANGE)]:
    band = pd.cut(d.epay_received_pct, bands, right=False, labels=labels)
    s = d.groupby(band, observed=True).collateral_required.agg(["mean", "count"])
    ax.plot(range(len(s)), 100 * s["mean"], marker="o", color=col, lw=2, ms=8, label=name)
    ax.annotate(name, (len(s) - 1, 100 * s["mean"].iloc[-1]), xytext=(8, 0),
                textcoords="offset points", va="center", color=INK2, fontsize=9)
ax.set_xticks(range(len(labels)), labels); ax.set_ylim(0, 100)
ax.set_xlabel("Share of sales received electronically (%)"); ax.set_ylabel("Borrowers asked for collateral (%)")
title(ax, "RO1: Collateral incidence by e-payment intensity", "Unadjusted shares, RQ1 sample")
ax.legend(frameon=False, loc="lower left"); ax.set_xlim(-0.3, len(labels) + 0.6)
fig.tight_layout(); fig.savefig(FIG / "fig2_ro1_incidence_by_epay.png"); plt.close(fig)

# ================================================================ Figure 3 (RO1 identification)
# First stage: own e-payment share against peers' share (instrument relevance)
fs = rq1.copy()
fs["zbin"] = pd.qcut(fs.z, 20, duplicates="drop")
bs = fs.groupby("zbin", observed=True).agg(z=("z", "mean"), e=("epay", "mean"))
fsd = fs.dropna(subset=CTRL_VARS)
fit = sm.OLS(fsd.epay, design(fsd).assign(z=fsd.z)).fit(
    cov_type="cluster", cov_kwds={"groups": pd.factorize(fsd.iv_cell)[0]})
fig, ax = plt.subplots(figsize=(6.5, 4.8))
ax.scatter(100 * bs.z, 100 * bs.e, s=60, color=BLUE, edgecolor=SURF, linewidth=1.5, zorder=3)
ax.plot([0, 100], [0, 100], color=INK2, lw=1, ls="--")
ax.set_xlim(0, 100); ax.set_ylim(0, 100)
ax.set_xlabel("Peers' e-payment share, same country-sector-region (%)")
ax.set_ylabel("Firm's own e-payment share (%)")
F = (fit.params["z"] / fit.bse["z"]) ** 2
title(ax, "RO1–RO2 identification: first stage",
      f"20 equal-sized bins; coefficient {fit.params['z']:.2f}, cluster-robust F = {F:.0f}")
fig.tight_layout(); fig.savefig(FIG / "fig3_first_stage.png"); plt.close(fig)

# ================================================================ Figure 4 (RO1 + RO2 estimates)
rows = []
for samp, d1, d2 in [("All seven countries", rq1, rq2),
                     ("Excluding India", rq1[rq1.country != "India"], rq2[rq2.country != "India"])]:
    b, se, n = ols(d1, "collateral_required");  rows.append(("Incidence (RO1)", "OLS", samp, b, se, n))
    b, se, n, _ = iv(d1, "collateral_required"); rows.append(("Incidence (RO1)", "2SLS", samp, b, se, n))
    b, se, n, _ = iv(d1, "collateral_required", FE_ALT); rows.append(("Incidence (RO1)", "2SLS alt FE", samp, b, se, n))
    b, se, n = ols(d2, "ln_collateral_to_loan");  rows.append(("Collateral/loan, log (RO2)", "OLS", samp, b, se, n))
    b, se, n, _ = iv(d2, "ln_collateral_to_loan"); rows.append(("Collateral/loan, log (RO2)", "2SLS", samp, b, se, n))
    b, se, n, _ = iv(d2, "ln_collateral_to_loan", FE_ALT); rows.append(("Collateral/loan, log (RO2)", "2SLS alt FE", samp, b, se, n))
est = pd.DataFrame(rows, columns=["outcome", "model", "sample", "coef", "se", "n"])
est.to_csv(FIG / "fig4_estimates.csv", index=False)
fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
for ax, (outcome, g) in zip(axes, est.groupby("outcome", sort=False)):
    yl = []
    for i, r in enumerate(g.itertuples()):
        col = BLUE if r.sample.startswith("All") else ORANGE
        ax.errorbar(r.coef, i, xerr=1.96 * r.se, fmt="o", color=col, ecolor=col, elinewidth=2, ms=8, capsize=0)
        yl.append(f"{r.model} · {r.sample} (n={r.n:,})")
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(range(len(yl)), yl); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Effect of 0% → 100% e-payments")
    title(ax, outcome, "Point estimate and 95% CI")
fig.text(0.01, 0.01, "2SLS: country×sector fixed effects. Alt FE: country×region fixed effects. "
         "SEs clustered by instrument cell.", fontsize=8.5, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(FIG / "fig4_ro1_ro2_estimates.png"); plt.close(fig)

# ================================================================ Figure 5 (RO2 composition)
types = {"coll_land_buildings": "Land &\nbuildings", "coll_machinery_equipment": "Machinery &\nequipment",
         "coll_receivables_inventory": "Receivables &\ninventory", "coll_personal_assets": "Personal\nassets",
         "coll_other": "Other"}
p = df[df.collateral_required == 1].dropna(subset=["epay_received_pct"]).copy()
p["tercile"] = pd.qcut(p.epay_received_pct.rank(method="first"), 3, labels=["Low", "Middle", "High"])
share = p.groupby("tercile", observed=True)[list(types)].mean() * 100
fig, ax = plt.subplots(figsize=(9, 4.6))
x, w = np.arange(len(types)), 0.26
for i, (t, row) in enumerate(share.iterrows()):
    ax.bar(x + (i - 1) * w, row.values, width=w - 0.02, color=TERCILE[i], label=f"{t} e-payment tercile")
ax.set_xticks(x, list(types.values())); ax.set_ylim(0, 100); ax.set_ylabel("% of pledging borrowers")
ax.grid(axis="x", visible=False)
title(ax, "RO2: Type of collateral pledged, by e-payment tercile",
      f"Borrowers who pledged collateral (n={len(p):,}); a loan can carry several types")
ax.legend(frameon=False, ncol=3, loc="upper right")
fig.tight_layout(); fig.savefig(FIG / "fig5_ro2_collateral_types.png"); plt.close(fig)

# ================================================================ Figure 6 (RO3 heterogeneity)
cint = df[df.collateral_required.notna()].groupby("country").apply(
    lambda g: wmean(g.collateral_required, g.weight_strict))
high = cint[cint >= cint.median()].index
splits = {
    "Firm age": [("Young (≤10 years)", lambda d: d.firm_age <= 10), ("Older (>10 years)", lambda d: d.firm_age > 10)],
    "Top manager": [("Woman", lambda d: d.woman_top_manager == 1), ("Man", lambda d: d.woman_top_manager == 0)],
    "Sector (tangibility proxy)": [("Manufacturing", lambda d: d.questionnaire_code == 1),
                                   ("Services", lambda d: d.questionnaire_code != 1)],
    "Country collateral intensity": [("High", lambda d: d.country.isin(high)), ("Low", lambda d: ~d.country.isin(high))],
}
het = []
for dim, groups in splits.items():
    for lab, cond in groups:
        for outcome, data in [("Incidence (RO1)", rq1), ("Collateral/loan, log (RO2)", rq2)]:
            d = data[cond(data)]
            y = "collateral_required" if outcome.startswith("Inc") else "ln_collateral_to_loan"
            try:
                b, se, n, f = iv(d, y)
            except Exception:
                b = se = f = np.nan; n = len(d)
            het.append((outcome, dim, lab, b, se, n, f))
het = pd.DataFrame(het, columns=["outcome", "dimension", "group", "coef", "se", "n", "first_stage_F"])
het.to_csv(FIG / "fig6_heterogeneity.csv", index=False)
fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
for ax, (outcome, g) in zip(axes, het.groupby("outcome", sort=False)):
    strong = g[g.first_stage_F >= 10]
    lo = (strong.coef - 1.96 * strong.se).min(); hi = (strong.coef + 1.96 * strong.se).max()
    pad = 0.15 * (hi - lo)
    ax.set_xlim(min(lo, 0) - pad, max(hi, 0) + pad)
    for i, r in enumerate(g.itertuples()):
        if r.first_stage_F >= 10:
            ax.errorbar(r.coef, i, xerr=1.96 * r.se, fmt="o", color=BLUE, ecolor=BLUE,
                        elinewidth=2, ms=8, capsize=0)
        else:   # weak first stage: estimate not interpretable, so it is not drawn
            ax.text(0.02, i, f"  weak instrument (F = {r.first_stage_F:.1f}), not shown",
                    transform=ax.get_yaxis_transform(), va="center", fontsize=8.5, color=INK2, style="italic")
    ax.axvline(0, color=INK2, lw=1); ax.grid(axis="y", visible=False)
    ax.set_yticks(range(len(g)), [f"{r.dimension}: {r.group} (n={r.n:,})" for r in g.itertuples()])
    ax.set_xlabel("Effect of 0% → 100% e-payments")
    ax.set_ylim(len(g) - 0.4, -0.6)
    title(ax, outcome, "2SLS point estimate and 95% CI")
fig.text(0.01, 0.01, "Country×sector fixed effects; SEs clustered by instrument cell. Groups with "
         "first-stage F < 10 are not drawn because their estimates are not reliable.", fontsize=8.5, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(FIG / "fig6_ro3_heterogeneity.png"); plt.close(fig)

print(est.round(3).to_string(index=False))
print(het.round(3).to_string(index=False))
print("Figures saved to", FIG.resolve())
