"""Task T6 cohort: confirmed COVID-19 in adults, Mexican Ministry of Health open registry (2020-2022), federated over real institutions.
  * patients: age >= 18, CLASIFICACION_FINAL 1 (clinical-epidemiological association) or 3 (laboratory/antigen confirmed); class 2 (confirmed only by the
    death-review committee) is excluded because it is defined by the outcome;
  * outcome: death within 30 days of the admission date (death date recorded in the registry);
  * predictors known at presentation: age, sex, nine chronic comorbidities (diabetes, COPD, asthma, immunosuppression, hypertension, other, cardiovascular disease,
    obesity, chronic renal disease), tobacco use and pregnancy. Hospital admission at registration is saved in the table but NOT used as a predictor: the registry updates
    it after the course of the illness, so it leaks the outcome (AUROC 0.96 alone);
  * institution = health-system sector (SSA, IMSS, ISSSTE, private, other public) x state of the care unit; clients are the cells with enough patients;
  * forward-in-time periods by admission date: train <= 2021-03-31, validation 2021-04-01 to 2021-06-30, test >= 2021-07-01 (Delta and Omicron waves, vaccination);
  * random subsample per (sector, state, period) cell to bound the compute (caps below).
Reads data/mx/COVID19MEXICO{2020,2021,2022}.zip, writes data/mx_processed.csv.gz.   Usage: python -I code/mx_prep.py   (stdlib, numpy, pandas only)"""
import os, sys, zipfile, json
import numpy as np, pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SRC = os.path.join(ROOT, "data", "mx"); OUT = os.path.join(ROOT, "data", "mx_processed.csv.gz")
COLS = ["SECTOR", "ENTIDAD_UM", "SEXO", "TIPO_PACIENTE", "FECHA_INGRESO", "FECHA_DEF", "EDAD", "EMBARAZO", "DIABETES", "EPOC", "ASMA", "INMUSUPR", "HIPERTENSION",
        "OTRA_COM", "CARDIOVASCULAR", "OBESIDAD", "RENAL_CRONICA", "TABAQUISMO", "CLASIFICACION_FINAL"]
COM = ["DIABETES", "EPOC", "ASMA", "INMUSUPR", "HIPERTENSION", "OTRA_COM", "CARDIOVASCULAR", "OBESIDAD", "RENAL_CRONICA", "TABAQUISMO"]
GROUP = {12: 0, 4: 1, 6: 2, 9: 3}                      # SSA, IMSS, ISSSTE, private; every other valid sector -> 4 (other public)
CAP = {"train": 1500, "val": 500, "test": 800}


def period(d):
    return np.where(d <= pd.Timestamp("2021-03-31"), "train", np.where(d <= pd.Timestamp("2021-06-30"), "val", "test"))


def main():
    parts = []; seen = 0
    for y in (2020, 2021, 2022):
        with zipfile.ZipFile(os.path.join(SRC, f"COVID19MEXICO{y}.zip")) as z:
            with z.open(z.namelist()[0]) as f:
                for ch in pd.read_csv(f, usecols=COLS, dtype=str, chunksize=500000, encoding="utf-8", low_memory=False):
                    seen += len(ch)
                    ch = ch[ch.CLASIFICACION_FINAL.isin(["1", "3"])]
                    for c in ["SECTOR", "ENTIDAD_UM", "SEXO", "TIPO_PACIENTE", "EDAD", "EMBARAZO"] + COM:
                        ch[c] = pd.to_numeric(ch[c], errors="coerce")
                    ch = ch[(ch.EDAD >= 18) & (ch.EDAD <= 110) & ch.SEXO.isin([1, 2]) & ch.TIPO_PACIENTE.isin([1, 2]) & ch.ENTIDAD_UM.between(1, 32) & ch.SECTOR.notna() & (ch.SECTOR != 99)].copy()
                    ch["adm"] = pd.to_datetime(ch.FECHA_INGRESO, errors="coerce"); ch["dead"] = pd.to_datetime(ch.FECHA_DEF.where(ch.FECHA_DEF != "9999-99-99"), errors="coerce")
                    ch = ch[ch.adm.notna() & (ch.adm >= pd.Timestamp("2020-01-01"))]
                    days = (ch.dead - ch.adm).dt.days
                    ch = ch[~(days < 0)]                                                          # death before admission: inconsistent record
                    days = (ch.dead - ch.adm).dt.days
                    out = pd.DataFrame({"sector": ch.SECTOR.map(lambda s: GROUP.get(int(s), 4)).astype(np.int8), "state": ch.ENTIDAD_UM.astype(np.int8), "age": ch.EDAD.astype(np.int16),
                                        "sex": (ch.SEXO == 2).astype(np.int8), "hosp": (ch.TIPO_PACIENTE == 2).astype(np.int8), "preg": (ch.EMBARAZO == 1).astype(np.int8),
                                        "period": period(ch.adm), "adm": ch.adm.dt.strftime("%Y-%m-%d"), "y": ((days >= 0) & (days <= 30)).astype(np.int8)})
                    unk = np.zeros(len(ch), dtype=np.int8)
                    for c in COM:
                        out[c.lower()] = (ch[c] == 1).astype(np.int8).values; unk += ch[c].isin([98, 99]).values.astype(np.int8)
                    out["n_unknown"] = unk; parts.append(out)
            print(y, "rows read so far", seen, "kept", sum(len(p) for p in parts), file=sys.stderr, flush=True)
    df = pd.concat(parts, ignore_index=True); n_all = len(df)
    rng = np.random.RandomState(0); keep = []
    for (s, st, p), g in df.groupby(["sector", "state", "period"]):
        cap = CAP[p]; keep.append(g if len(g) <= cap else g.iloc[rng.permutation(len(g))[:cap]])
    d = pd.concat(keep, ignore_index=True)
    d.to_csv(OUT, index=False, compression="gzip")
    cnt = d.groupby("period").agg(n=("y", "size"), deaths=("y", "sum")); cnt["rate"] = (cnt.deaths / cnt.n).round(4)
    full = df.groupby("period").agg(n=("y", "size"), deaths=("y", "sum"))
    json.dump(dict(rows_read=int(seen), eligible_adults=int(n_all), eligible_by_period={k: dict(n=int(v.n), deaths=int(v.deaths)) for k, v in full.iterrows()},
                   sample_by_period={k: dict(n=int(v.n), deaths=int(v.deaths)) for k, v in cnt.iterrows()}), open(os.path.join(ROOT, "results", "mx_cohort.json"), "w"), indent=1)
    print(cnt); print(d.groupby("sector").agg(n=("y", "size"), rate=("y", "mean")).round(3)); print("cells", d.groupby(["sector", "state"]).ngroups)


if __name__ == "__main__":
    main()
