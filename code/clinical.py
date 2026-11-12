"""Clinical-style comparator models: small pooled logistic regressions on the few predictors a clinician would use.
These are *refitted* models inspired by classic scores (not the published coefficient tables):
  T1  pretest-probability style : age, sex, chest-pain type            (Diamond-Forrester-type)
  T2  FINDRISC-style            : age, BMI, physical activity, fruit, vegetables, high BP
  T3  simple risk factors       : age, sex, BMI, smoking, diabetes, physical activity
Usage: python clinical.py T1|T2|T3"""
import sys, os
import numpy as np, pandas as pd
import data, fl
from metrics import full_metrics, youden_threshold, auroc
from settings import SET

RES = os.environ.get("HIPERC_RESULTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
COLS = {"T1": ["age", "sex", "cp=1", "cp=2", "cp=3", "cp=4"],
        "T2": ["Age", "BMI", "PhysActivity", "Fruits", "Veggies", "HighBP"],
        "T3": ["Age", "Sex", "BMI", "Smoker", "Diabetes_binary", "PhysActivity"],
        "T4": ["age", "bmi", "waist", "sbp"],
        "T5": ["age", "HR_last", "Temp_last", "Resp_last", "WBC_last"],
        "T6": ["age", "sex"],                                                         # age and sex only
        "T5H24": ["age", "HR_last", "Temp_last", "Resp_last", "WBC_last"]}               # SIRS-style: heart rate, temperature, respiration, white cells


def main(task):
    rows = []
    for seed in SET[task]["seeds"]:
        fed, _, info = data.build(task, seed)
        cols = [info["names"].index(c) for c in COLS[task]]
        r = fl.run_logreg(fed, X_cols=cols)
        yv = np.concatenate([c["yva"].numpy() for c in fed]); pv = np.concatenate(r["val"])
        y = np.concatenate([c["yte"].numpy() for c in fed]); p = np.concatenate(r["test"])
        thr = youden_threshold(yv, pv)
        row = dict(task=task, spec="clinical_lr", method="clinical_lr", seed=seed, best_round=0, val_auc=r["val_auc"],
                   wan_MB=0, lan_MB=0, secs=0, **full_metrics(y, p, thr))
        hosp = np.concatenate([c["meta_te"]["hosp"] for c in fed])
        for h in range(info["n_hosp"]):
            row[f"auroc_h{h}"] = auroc(y[hosp == h], p[hosp == h])
        meta = {k: np.concatenate([c["meta_te"][k] for c in fed]) for k in fed[0]["meta_te"]}
        out = os.path.join(RES, "preds", f"{task}_main"); os.makedirs(out, exist_ok=True)
        np.savez_compressed(os.path.join(out, f"clinical_lr_s{seed}.npz"), y=y.astype(np.int8), p=p.astype(np.float32),
                            yv=yv.astype(np.int8), pv=pv.astype(np.float32), thr=thr, curve=np.zeros((0, 4)), **meta)
        rows.append(row)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, f"runs_{task}_clinical.csv"), index=False)
    print(task, "clinical LR AUROC %.3f +- %.3f" % (df.auroc.mean(), df.auroc.std()))


if __name__ == "__main__":
    main(sys.argv[1])
