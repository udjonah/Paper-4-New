"""
03_export_excel.py
Writes the clean dataset, a codebook and a formula-driven country summary
to output/WBES_epay_collateral_clean_data.xlsx
"""
from pathlib import Path
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

df = pd.read_csv("output/wbes_epay_collateral_clean.csv")
OUT = Path("output/WBES_epay_collateral_clean_data.xlsx")

CODEBOOK = [
    # variable, description, WBES source, coding
    ("country", "Country", "file", "Text"),
    ("wave", "Survey wave year", "file", "Year"),
    ("idstd", "WBES establishment ID", "idstd", "ID"),
    ("weight_strict", "Sampling weight (strict eligibility)", "wstrict", "Use for population estimates"),
    ("strata", "Sampling stratum", "strata", "Code"),
    ("sector_code / sector", "Industry sampling sector", "a4a", "Code and value label"),
    ("region_code / region", "Region of the establishment", "a3a", "Code and value label"),
    ("size_class_code / size_class", "Sampling size class", "a6a", "Small / Medium / Large"),
    ("questionnaire_code / questionnaire", "Questionnaire type", "a0", "1 = Manufacturing; 2-3 = services"),
    ("employees", "Permanent full-time employees, end of last fiscal year", "l1", "Count"),
    ("firm_age", "Wave year minus year operations began", "b5", "Years"),
    ("woman_top_manager", "Top manager is a woman", "b7a", "1 = Yes, 0 = No"),
    ("woman_owner", "Any woman among the owners", "b4", "1 = Yes, 0 = No"),
    ("foreign_owned", "Foreign ownership of 10% or more", "b2b", "1/0"),
    ("exporter", "Direct + indirect exports of 10% of sales or more", "d3b + d3c", "1/0"),
    ("audited", "Financial statements externally audited", "k21", "1 = Yes, 0 = No"),
    ("website", "Establishment has its own website", "c22b", "1 = Yes, 0 = No"),
    ("has_loan_or_line", "Has a loan or line of credit from a financial institution", "k82a/k82b, k82 or k8 (varies by wave)", "1 = Yes, 0 = No"),
    ("applied_loan", "Applied for a loan in last fiscal year", "k162a", "1 = Yes, 0 = No; blank where not asked"),
    ("loan_outcome_code", "Outcome of most recent application", "k20a1", "1 approved in full, 2 in part, 3 rejected, 4 withdrawn"),
    ("collateral_required", "Most recent loan/line of credit required collateral", "k13", "1 = Yes, 0 = No. Collateral types (k14a-e) are asked only when k13 = Yes"),
    ("loan_value_lcu", "Value of most recent loan at approval", "k11", "Local currency units"),
    ("collateral_value_lcu", "Value of collateral required", "k15a", "Local currency units"),
    ("collateral_to_loan_pct", "Collateral value / loan value x 100", "k15a / k11", "Percent; unitless across countries"),
    ("ln_collateral_to_loan", "Natural log of collateral/loan", "k15a / k11", "RO2 outcome"),
    ("coll_land_buildings", "Collateral type: land, buildings", "k14a", "1/0, only if collateral_required = 1"),
    ("coll_machinery_equipment", "Collateral type: machinery and equipment", "k14b", "1/0, only if collateral_required = 1"),
    ("coll_receivables_inventory", "Collateral type: accounts receivable and inventories", "k14c", "1/0, only if collateral_required = 1"),
    ("coll_personal_assets", "Collateral type: personal assets of owner", "k14d", "1/0, only if collateral_required = 1"),
    ("coll_other", "Collateral type: other", "k14e", "1/0, only if collateral_required = 1"),
    ("days_to_loan_decision", "Days to receive decision on most recent application", "k32", "Days"),
    ("epay_received_pct", "Share of payments received electronically", "k33", "Percent (0-100); main explanatory variable"),
    ("epay_made_pct", "Share of payments made electronically", "k38", "Percent (0-100)"),
    ("epay_receive_cost_pct", "Cost per transaction of main e-payment received", "k36", "Percent of transaction"),
    ("iv_cell", "Instrument cell: country | sector | region", "derived", "Text"),
    ("iv_cell_n", "Firms in the cell reporting k33", "derived", "Count"),
    ("iv_loo_epay_pct", "Instrument: mean e-payment share of other firms in the cell", "derived from k33", "Percent; blank if fewer than 6 firms in cell"),
    ("ln_employees", "ln(1 + employees)", "derived", "Control"),
    ("ln_firm_age", "ln(1 + firm_age)", "derived", "Control"),
    ("in_rq1_sample", "Row is in the RO1 estimation sample", "derived", "1/0: non-missing collateral_required, e-payment, instrument and core controls"),
    ("in_rq2_sample", "Row is in the RO2 estimation sample", "derived", "1/0: RO1 sample with a valid collateral/loan ratio"),
]

ARIAL = "Arial"
HEAD_FILL = PatternFill("solid", start_color="1F3864")
HEAD_FONT = Font(name=ARIAL, bold=True, color="FFFFFF")
BODY = Font(name=ARIAL, size=10)


def header(ws, row=1):
    for c in ws[row]:
        c.font, c.fill = HEAD_FONT, HEAD_FILL
        c.alignment = Alignment(wrap_text=True, vertical="center")


wb = Workbook()

# ---------------------------------------------------------------- README
ws = wb.active; ws.title = "README"
lines = [
    ("WBES electronic payments and collateral: clean analysis dataset", True),
    ("Source: World Bank Enterprise Surveys, latest standard wave per country: India 2025, Kenya 2025, "
     "Nigeria 2025, Poland 2025, Türkiye 2024, Mexico 2023, Vietnam 2023 (https://www.enterprisesurveys.org).", False),
    (f"Rows: {len(df):,} establishments. One row per firm. Built by 01_build_clean_data.py.", False),
    ("Cleaning rules: WBES negative codes (don't know, refused, not applicable) set to blank; "
     "yes/no items recoded 1 = Yes, 0 = No; monetary values kept in local currency.", False),
    ("Collateral questions (k11, k13, k14, k15) refer to the firm's most recent loan or line of credit "
     "and are answered only by firms that have one.", False),
    ("Population estimates should use weight_strict. Regression samples are flagged by in_rq1_sample "
     "and in_rq2_sample.", False),
    ("Sheets: Clean_Data (the dataset), Codebook (variable definitions and WBES codes), "
     "Country_Summary (live formulas over Clean_Data).", False),
]
for i, (t, bold) in enumerate(lines, start=1):
    c = ws.cell(row=i, column=1, value=t); c.font = Font(name=ARIAL, bold=bold, size=12 if bold else 10)
    c.alignment = Alignment(wrap_text=True)
ws.column_dimensions["A"].width = 120

# ---------------------------------------------------------------- Clean_Data
wd = wb.create_sheet("Clean_Data")
cols = list(df.columns)
wd.append(cols)
for row in df.itertuples(index=False):
    wd.append([None if (isinstance(v, float) and pd.isna(v)) else v for v in row])
header(wd)
wd.freeze_panes = "B2"
for j, c in enumerate(cols, start=1):
    wd.column_dimensions[get_column_letter(j)].width = max(12, min(28, len(c) + 2))
for row in wd.iter_rows(min_row=2):
    for c in row:
        c.font = BODY
pct_cols = ["collateral_to_loan_pct", "epay_received_pct", "epay_made_pct", "iv_loo_epay_pct"]
for name in pct_cols + ["ln_collateral_to_loan", "ln_employees", "ln_firm_age", "weight_strict"]:
    j = cols.index(name) + 1
    for (c,) in wd.iter_rows(min_row=2, min_col=j, max_col=j):
        c.number_format = "0.00"
wd.auto_filter.ref = wd.dimensions

# ---------------------------------------------------------------- Codebook
wc = wb.create_sheet("Codebook")
wc.append(["Variable", "Description", "WBES source", "Coding / notes"])
for r in CODEBOOK:
    wc.append(list(r))
header(wc)
for row in wc.iter_rows(min_row=2):
    for c in row:
        c.font = BODY; c.alignment = Alignment(wrap_text=True, vertical="top")
for col, w in zip("ABCD", [32, 55, 30, 60]):
    wc.column_dimensions[col].width = w
wc.freeze_panes = "A2"

# ---------------------------------------------------------------- Country_Summary (formulas)
wsum = wb.create_sheet("Country_Summary")
n = len(df) + 1
L = {c: get_column_letter(cols.index(c) + 1) for c in cols}
rng = lambda c: f"Clean_Data!${L[c]}$2:${L[c]}${n}"
heads = ["Country", "Firms", "Borrowers answering collateral question", "RO1 sample (n)", "RO2 sample (n)",
         "Collateral required, unweighted (%)", "Collateral required, survey-weighted (%)",
         "Mean collateral/loan (%)", "Mean e-payment share received (%)"]
wsum.append(heads)
countries = sorted(df.country.unique())
for i, c in enumerate(countries, start=2):
    C, K, W = rng("country"), rng("collateral_required"), rng("weight_strict")
    wsum.append([
        c,
        f'=COUNTIF({C},A{i})',
        f'=COUNTIFS({C},A{i},{K},"<>")',
        f'=SUMIFS({rng("in_rq1_sample")},{C},A{i})',
        f'=SUMIFS({rng("in_rq2_sample")},{C},A{i})',
        f'=AVERAGEIFS({K},{C},A{i})',
        f'=SUMPRODUCT(({C}=A{i})*({K}<>"")*{W},{K})/SUMPRODUCT(({C}=A{i})*({K}<>"")*{W})',
        f'=AVERAGEIFS({rng("collateral_to_loan_pct")},{C},A{i})/100',
        f'=AVERAGEIFS({rng("epay_received_pct")},{C},A{i})/100',
    ])
last = len(countries) + 1
t = last + 1
wsum.append(["All seven", f"=SUM(B2:B{last})", f"=SUM(C2:C{last})", f"=SUM(D2:D{last})", f"=SUM(E2:E{last})",
             f'=AVERAGEIFS({rng("collateral_required")},{rng("collateral_required")},"<>")', "",
             f'=AVERAGE({rng("collateral_to_loan_pct")})/100', f'=AVERAGE({rng("epay_received_pct")})/100'])
header(wsum)
for row in wsum.iter_rows(min_row=2, max_row=t):
    for c in row:
        c.font = Font(name=ARIAL, size=10, bold=(c.row == t))
    for c in row[1:5]:
        c.number_format = "#,##0"
    for c in row[5:]:
        c.number_format = "0.0%"
wsum.cell(row=t + 2, column=1, value=(
    "Notes: all cells are formulas over Clean_Data. The weighted share uses weight_strict within each country, "
    "so country rows are comparable but the 'All seven' row is left unweighted. Mean collateral/loan is "
    "pulled up by a few very large pledges; the median is reported in Figure 1 of the analysis.")).font = Font(name=ARIAL, size=9, italic=True)
for col, w in zip("ABCDEFGHI", [14, 9, 18, 12, 12, 16, 16, 14, 16]):
    wsum.column_dimensions[col].width = w
wsum.row_dimensions[1].height = 45
wsum.freeze_panes = "B2"

wb.move_sheet("Country_Summary", offset=-1)
wb.save(OUT)
print("saved", OUT)
