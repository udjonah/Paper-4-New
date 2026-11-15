"""
01_build_clean_data.py
Builds the analysis dataset for:
"Instrumental Variable Evidence on Electronic Payment Trails Reducing
 Collateral Dependence in SME Lending across Emerging Economies"

Source: World Bank Enterprise Surveys (WBES), latest standard wave per country.
Run on Google Colab or locally:  python 01_build_clean_data.py
Requires: pandas, numpy, pyreadstat
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pyreadstat

# ---------------------------------------------------------------------------
# 1. Paths: edit DATA_DIR to the folder where you unzipped the WBES downloads
# ---------------------------------------------------------------------------
DATA_DIR = Path("wbes_data")
OUT_DIR = Path("output")
OUT_DIR.mkdir(exist_ok=True)

FILES = {  # country: (relative path to .dta, survey wave year)
    "India":   ("India-2025-full-data.dta/India-2025-full-data.dta", 2025),
    "Kenya":   ("WBES_Kenya2025_Data/Kenya-2025-full-data.dta", 2025),
    "Nigeria": ("WBES_Nigeria2025_Data/Nigeria-2025-full-data.dta", 2025),
    "Poland":  ("WBES_Poland2025_Data/Poland-2025-full-data.dta", 2025),
    "Turkiye": ("WBES_Turkiye2024_Data/Turkiye-2024-full-data.dta", 2024),
    "Mexico":  ("WBES_Mexico2023_Data/Mexico-2023-full-data.dta", 2023),
    "Vietnam": ("WBES_Vietnam2023_Data/Viet-Nam-2023-full-data.dta", 2023),
}

# Raw WBES variables used (questionnaire codes)
RAW = ["idstd", "wstrict", "strata", "a0", "a3a", "a4a", "a6a", "b2b", "b4", "b5",
       "b7a", "d3b", "d3c", "k21", "c22b", "l1",
       "k8", "k82", "k82a", "k82b", "k162a", "k20a1",
       "k11", "k13", "k14a", "k14b", "k14c", "k14d", "k14e", "k15a", "k32",
       "k33", "k36", "k38"]


def yes_no(s):
    """WBES yes/no items: 1 = Yes, 2 = No, negatives = don't know/refused."""
    return pd.Series(np.select([s == 1, s == 2], [1.0, 0.0], np.nan), index=s.index)


def load_country(country, rel_path, wave):
    df, meta = pyreadstat.read_dta(DATA_DIR / rel_path, encoding="latin1")
    keep = [v for v in RAW if v in df.columns]
    df = df[keep].copy()
    labels = meta.variable_value_labels
    out = pd.DataFrame(index=df.index)
    out["country"] = country
    out["wave"] = wave
    out["idstd"] = df["idstd"]
    out["weight_strict"] = df["wstrict"]
    out["strata"] = df["strata"]

    # Value labels -> text, so sector/region names are explicit
    for raw, new in [("a4a", "sector"), ("a3a", "region"), ("a6a", "size_class"), ("a0", "questionnaire")]:
        lab = labels.get(raw, {})
        out[new + "_code"] = df[raw]
        out[new] = df[raw].map(lab)

    num = df.drop(columns=["idstd"]).apply(pd.to_numeric, errors="coerce")
    num = num.mask(num < 0)  # negative codes = don't know / refused / not applicable

    # Firm characteristics
    out["employees"] = num.get("l1")
    out["firm_age"] = (wave - num.get("b5")).where(lambda s: s >= 0)
    out["woman_top_manager"] = yes_no(num["b7a"])
    out["woman_owner"] = yes_no(num["b4"])
    out["foreign_owned"] = (num["b2b"] >= 10).astype(float).where(num["b2b"].notna())
    exp_share = num[["d3b", "d3c"]].sum(axis=1, min_count=1)
    out["exporter"] = (exp_share >= 10).astype(float).where(exp_share.notna())
    out["audited"] = yes_no(num["k21"])
    out["website"] = yes_no(num["c22b"])

    # Has a loan or credit line: variable name differs by wave
    if "k82a" in num:
        has = yes_no(num["k82a"])
        if "k82b" in num:
            has = pd.concat([has, yes_no(num["k82b"])], axis=1).max(axis=1, skipna=True)
    elif "k82" in num:
        has = yes_no(num["k82"])
    else:
        has = yes_no(num["k8"])
    out["has_loan_or_line"] = has
    out["applied_loan"] = yes_no(num["k162a"]) if "k162a" in num else np.nan
    out["loan_outcome_code"] = df.get("k20a1").where(df.get("k20a1") > 0) if "k20a1" in df else np.nan

    # Most recent loan: collateral (k13 asked of firms with a loan/credit line;
    # k14a-e and k15a are asked only if k13 = Yes)
    out["collateral_required"] = yes_no(num["k13"])
    out["loan_value_lcu"] = num["k11"].where(num["k11"] > 0)
    out["collateral_value_lcu"] = num["k15a"].where(num["k15a"] > 0)
    out["collateral_to_loan_pct"] = 100 * out["collateral_value_lcu"] / out["loan_value_lcu"]
    out["ln_collateral_to_loan"] = np.log(out["collateral_to_loan_pct"] / 100)
    for raw, new in [("k14a", "coll_land_buildings"), ("k14b", "coll_machinery_equipment"),
                     ("k14c", "coll_receivables_inventory"), ("k14d", "coll_personal_assets"),
                     ("k14e", "coll_other")]:
        out[new] = yes_no(num[raw]).where(out["collateral_required"] == 1)
    out["days_to_loan_decision"] = num.get("k32")

    # Electronic payments
    out["epay_received_pct"] = num["k33"]
    out["epay_made_pct"] = num.get("k38")
    out["epay_receive_cost_pct"] = num.get("k36")
    return out


def main():
    df = pd.concat([load_country(c, *v) for c, v in FILES.items()], ignore_index=True)

    # Instrument: leave-one-out mean of e-payment share in country x sector x region cell
    df["iv_cell"] = df["country"] + " | " + df["sector"].astype(str) + " | " + df["region"].astype(str)
    g = df.groupby("iv_cell")["epay_received_pct"]
    tot, n = g.transform("sum"), g.transform("count")
    own = df["epay_received_pct"]
    df["iv_cell_n"] = n
    df["iv_loo_epay_pct"] = np.where(own.notna(), (tot - own) / (n - 1), tot / n)
    df.loc[df["iv_cell_n"] < 6, "iv_loo_epay_pct"] = np.nan  # too few peers

    df["ln_employees"] = np.log1p(df["employees"])
    df["ln_firm_age"] = np.log1p(df["firm_age"])

    core = ["collateral_required", "epay_received_pct", "iv_loo_epay_pct",
            "ln_employees", "ln_firm_age", "woman_top_manager", "audited", "website"]
    df["in_rq1_sample"] = df[core].notna().all(axis=1).astype(int)
    df["in_rq2_sample"] = ((df["in_rq1_sample"] == 1) &
                           df["ln_collateral_to_loan"].notna()).astype(int)

    df.to_csv(OUT_DIR / "wbes_epay_collateral_clean.csv", index=False)
    print(df.groupby("country").agg(firms=("idstd", "size"),
                                    rq1=("in_rq1_sample", "sum"),
                                    rq2=("in_rq2_sample", "sum")))
    return df


if __name__ == "__main__":
    main()
