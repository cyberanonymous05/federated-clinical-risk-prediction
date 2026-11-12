"""Build the NHANES 2005-2018 adult diabetes cohort (7 two-year survey cycles) from the public CDC files in data/nhanes.
Label  : diabetes = doctor-diagnosed (DIQ010 == 1) OR HbA1c >= 6.5% (LBXGH) OR fasting glucose >= 126 mg/dL (LBXGLU);
         participants need a measured HbA1c, or a diagnosis, to be included (so undiagnosed diabetes can be detected).
Inputs : age, sex, race/ethnicity, family income-to-poverty ratio, BMI, waist circumference, mean systolic and diastolic blood
         pressure (measured), total and HDL cholesterol. HbA1c, glucose and the diabetes questionnaire are used ONLY for the label.
Output : data/nhanes_processed.csv   Usage: python nhanes_prep.py"""
import os, numpy as np, pandas as pd
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
CYC = [("D", "2005-2006"), ("E", "2007-2008"), ("F", "2009-2010"), ("G", "2011-2012"), ("H", "2013-2014"), ("I", "2015-2016"), ("J", "2017-2018")]


def rd(t, s):
    return pd.read_sas(os.path.join(D, "nhanes", f"{t}_{s}.xpt"), format="xport", encoding="utf-8")


rows = []
for k, (s, name) in enumerate(CYC):
    demo = rd("DEMO", s)[["SEQN", "RIAGENDR", "RIDAGEYR", "RIDRETH1", "INDFMPIR"]]
    bmx = rd("BMX", s)[["SEQN", "BMXBMI", "BMXWAIST"]]
    bpx = rd("BPX", s); sy = [c for c in bpx.columns if c.startswith("BPXSY")]; di = [c for c in bpx.columns if c.startswith("BPXDI")]
    bp = pd.DataFrame({"SEQN": bpx.SEQN, "sbp": bpx[sy].replace(0, np.nan).mean(axis=1), "dbp": bpx[di].replace(0, np.nan).mean(axis=1)})
    tc = rd("TCHOL", s)[["SEQN", "LBXTC"]]; hdl = rd("HDL", s); hdl = hdl[["SEQN", [c for c in hdl.columns if c.startswith("LBDHDD") or c == "LBDHDL"][0]]]
    hdl.columns = ["SEQN", "hdl"]
    ghb = rd("GHB", s)[["SEQN", "LBXGH"]]; glu = rd("GLU", s)[["SEQN", "LBXGLU"]]; diq = rd("DIQ", s)[["SEQN", "DIQ010"]]
    df = demo
    for o in (bmx, bp, tc, hdl, ghb, glu, diq):
        df = df.merge(o, on="SEQN", how="left")
    df["cycle"] = k
    rows.append(df)
df = pd.concat(rows, ignore_index=True)
df = df[df.RIDAGEYR >= 20].copy()
dx = df.DIQ010 == 1
lab = dx | (df.LBXGH >= 6.5) | (df.LBXGLU >= 126)
keep = df.LBXGH.notna() | dx
df = df[keep].copy(); df["y"] = lab[keep].astype(int)
out = pd.DataFrame({"cycle": df.cycle, "y": df.y, "age": df.RIDAGEYR, "sex": (df.RIAGENDR == 1).astype(int), "race": df.RIDRETH1,
                    "pir": df.INDFMPIR, "bmi": df.BMXBMI, "waist": df.BMXWAIST, "sbp": df.sbp, "dbp": df.dbp, "tc": df.LBXTC, "hdl": df.hdl})
out.to_csv(os.path.join(D, "nhanes_processed.csv"), index=False)
print("rows", len(out), "prevalence %.3f" % out.y.mean())
print(out.groupby("cycle").agg(n=("y", "size"), prev=("y", "mean"), age=("age", "mean"), bmi=("bmi", "mean"), sbp=("sbp", "mean")).round(3))
print(out.isna().mean().round(3).to_dict())
