"""Patient-level cohort for task T5 from the PhysioNet/Computing in Cardiology Challenge 2019 training data (two hospital systems).
Landmark design (no information from after the landmark enters a feature):
  * landmark = ICU hour 24; features come from ICU hours 1-24 only (last, min, max of each of the 34 time-varying variables, plus age, sex,
    hours from hospital to ICU admission, and the two ICU-unit flags);
  * eligible: stay reaches hour 24 and the sepsis label is still 0 at hour 24;
  * outcome y = 1 if the Challenge sepsis label turns to 1 at any later hour of the recorded stay (the Challenge label starts 6 h before the
    clinical onset; there is no fixed horizon and the length of the recorded stay differs between patients - a limitation we report).
Reads data/sepsis/training_setA|B/*.psv, writes data/sepsis_processed.csv.  Usage: python -I code/sepsis_prep.py   (stdlib, numpy, pandas only)"""
import os, glob, sys, json
import numpy as np, pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SRC = os.path.join(ROOT, "data", "sepsis"); OUT = os.path.join(ROOT, "data", "sepsis_processed.csv")
VARS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2", "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN", "Alkalinephos",
        "Calcium", "Chloride", "Creatinine", "Bilirubin_direct", "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium", "Bilirubin_total",
        "TroponinI", "Hct", "Hgb", "PTT", "WBC", "Fibrinogen", "Platelets"]
LM = 24


def one(path, system):
    d = pd.read_csv(path, sep="|")
    if d.ICULOS.max() < LM:
        return "short", None
    pre = d[d.ICULOS <= LM]
    if len(pre) == 0:
        return "no_early_rows", None
    if pre.SepsisLabel.max() > 0:
        return "already", None
    post = d[d.ICULOS > LM]
    win = d[(d.ICULOS > LM) & (d.ICULOS <= LM + 24)]            # fixed-horizon sensitivity: sepsis label within the next 24 hours
    y24 = int(win.SepsisLabel.max() > 0) if len(win) else 0
    elig24 = int(y24 == 1 or d.ICULOS.max() >= LM + 24)         # complete 24-hour follow-up, or the event occurred inside the window
    r = dict(pid=os.path.basename(path)[:-4], system=system, y=int(post.SepsisLabel.max() > 0) if len(post) else 0, n_after=len(post), y24=y24, elig24=elig24,
             age=pre.Age.iloc[0], sex=pre.Gender.iloc[0], hospadm=pre.HospAdmTime.iloc[0], unit1=pre.Unit1.iloc[0], unit2=pre.Unit2.iloc[0])
    for v in VARS:
        x = pre[v].dropna()
        r[v + "_last"], r[v + "_min"], r[v + "_max"] = (x.iloc[-1], x.min(), x.max()) if len(x) else (np.nan, np.nan, np.nan)
        r[v + "_n"] = len(x)
    return "ok", r


def main():
    rows = []; cnt = {}
    for k, s in (("training_setA", 0), ("training_setB", 1)):
        fs = sorted(glob.glob(os.path.join(SRC, k, "p*.psv")))
        for f in fs:
            st, r = one(f, s); cnt[(s, st)] = cnt.get((s, st), 0) + 1
            if r:
                rows.append(r)
        print(k, len(fs), "files", file=sys.stderr)
    df = pd.DataFrame(rows); df.to_csv(OUT, index=False)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    json.dump({f"{k[0]}|{k[1]}": v for k, v in cnt.items()}, open(os.path.join(ROOT, "results", "sepsis_cohort.json"), "w"), indent=1)
    print({k: v for k, v in sorted(cnt.items())}); print(df.groupby("system").y.agg(["size", "mean"]).round(4)); print(df.groupby(["system", "unit1", "unit2"], dropna=False).y.agg(["size", "mean"]).round(3))


if __name__ == "__main__":
    main()
