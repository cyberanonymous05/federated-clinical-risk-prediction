"""Companion to lr_sensitivity.py: where the validation-selected learning rate was the LOWEST value of the grid (hierarchical FedAvg on T1,
HiPerC-FL on T4), also try 0.01. Appends to results/lr_sensitivity.csv. Usage: python lr_sensitivity_low.py"""
import os, json
import pandas as pd
from multiprocessing import Pool
from lr_sensitivity import job, RES

if __name__ == "__main__":
    jobs = []
    for task, name in (("T1", "hierfavg"), ("T4", "hiperc")):
        best = json.load(open(os.path.join(RES, f"configs_{task}.json")))["best"]; cl = {k: v for k, v in best[name].items() if not k.startswith("_")}
        jobs += [(task, name, name, {**cl, "lr": 0.01}, s) for s in range(5)]
    with Pool(9) as p:
        rows = p.map(job, jobs, chunksize=1)
    f = os.path.join(RES, "lr_sensitivity.csv"); df = pd.concat([pd.read_csv(f), pd.DataFrame(rows)], ignore_index=True); df.to_csv(f, index=False)
    print(pd.DataFrame(rows).groupby(["task", "method", "lr"]).auroc.mean().round(4))
