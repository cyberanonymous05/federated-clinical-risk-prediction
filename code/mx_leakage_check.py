"""Leakage check for T6: which registry fields carry the outcome? Logistic regressions on the training period, AUROC in the later test period.
Hospital admission at registration is updated by the registry after the course of the illness; it must not be used as a predictor. Writes results/mx_leakage.json.  Usage: python mx_leakage_check.py"""
import os, json
import numpy as np, pandas as pd, statsmodels.api as sm
from metrics import auroc

RES = os.environ.get("HIPERC_RESULTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
df = pd.read_csv(os.path.join(RES, "..", "data", "mx_processed.csv.gz")); tr = df[df.period == "train"]; te = df[df.period == "test"]
base = ["age", "sex", "preg", "diabetes", "epoc", "asma", "inmusupr", "hipertension", "otra_com", "cardiovascular", "obesidad", "renal_cronica", "tabaquismo"]
out = {}
for name, fs in (("all_with_hosp_and_unknown", base + ["hosp", "n_unknown"]), ("hosp_alone", ["hosp"]), ("n_unknown_alone", ["n_unknown"]), ("baseline_predictors", base), ("age_sex", ["age", "sex"])):
    m = sm.Logit(tr.y, sm.add_constant(tr[fs].astype(float))).fit(disp=0); p = m.predict(sm.add_constant(te[fs].astype(float))); out[name] = round(float(auroc(te.y.values, p.values)), 3)
out["mortality_hosp1"] = round(float(tr[tr.hosp == 1].y.mean()), 3); out["mortality_hosp0"] = round(float(tr[tr.hosp == 0].y.mean()), 3)
json.dump(out, open(os.path.join(RES, "mx_leakage.json"), "w"), indent=1); print(out)
