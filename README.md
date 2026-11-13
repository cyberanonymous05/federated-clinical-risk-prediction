# Federated clinical risk prediction: code

Python code for training and evaluating federated learning methods for clinical risk prediction on six public data sets, and for checking the results. It contains the federated algorithms (FedAvg, FedProx, SCAFFOLD, Ditto, pFedMe, first-order Per-FedAvg, IFCA, hierarchical FedAvg, and HiPerC-FL, which combines two-tier aggregation, personal models, an invariance penalty and continual adaptation), pooled and linear references, pooled and federated gradient-boosted trees, differential privacy (DP-SGD), patient-level statistics, held-out-institution and forward-in-time validation, drift experiments, and audits for data leakage.

This is research code. It is not a clinical tool and has not been validated for use in care.

## Tasks and data
The public data are not redistributed here; scripts download them from their sources. File checksums are in `data/`.

| Task | Data | Source |
|---|---|---|
| T1 | Coronary disease, four hospitals (UCI Heart Disease, 920 patients) | UCI Machine Learning Repository, doi:10.24432/C52P4X (`download_data.sh`) |
| T2, T3 | Diabetes and high blood pressure, 253,680 adults (BRFSS 2015 indicators) | UCI Machine Learning Repository, doi:10.24432/C53919 (`download_data.sh`) |
| T4 | Diabetes, NHANES 2005 to 2018, 36,666 adults | CDC National Center for Health Statistics (`download_nhanes.py`, then `code/nhanes_prep.py`) |
| T5 | Sepsis in two hospital systems, PhysioNet/Computing in Cardiology Challenge 2019 | PhysioNet, doi:10.13026/v64v-d857 (`download_sepsis.py`, then `code/sepsis_prep.py`; the project page states CC BY 4.0 and the licence file in the download states ODbL 1.0) |
| T6 | Death within 30 days, confirmed COVID-19 in adults, Mexican open registry 2020 to 2022 | Secretaria de Salud, Mexico, open data (Libre Uso MX; `download_mx.py`, then `code/mx_prep.py`; about 300 MB) |

Check each data source's terms before use.

## Layout
- `code/` all Python code. `fl.py` algorithms; `data.py` task builders; `settings.py` rounds, seeds and tuning grids; `tune.py` and `run_main.py` tuning and main runs; `metrics.py` and `stats_patient.py` metrics and patient-level inference (DeLong, Holm); `loho*.py`, `nhanes_*.py`, `rolling_mx.py` held-out and forward-in-time validation; `drift.py` drift streams; `dp_exp.py` privacy; `gbm_exp.py` and `fedgbm.py` boosted trees; `hical.py` HiCal federated recalibration; `blend.py` blending of model classes; `leakage_audit.py`, `audit_independent.py` and `validity_checks.py` audits and implementation tests; `check_headlines.py` one-command check of headline results.
- `data/` SHA-256 checksums of the downloaded files.
- `results/` tuned configurations, per-run results and summary tables (saved model predictions are not included because of their size).
- `reproduce.sh`, `Dockerfile`, `requirements.txt`, `environment.yml` reproduction.

## Run
Docker (recommended):
```
docker build -t hiperc-fl .
docker run --rm hiperc-fl ./reproduce.sh smoke
```
The smoke test downloads the data for T1 to T4, runs T1 with two seeds and 20 rounds, and runs the implementation tests, the leakage audit and the boosted-tree reference. It takes minutes and prints `SMOKE TEST PASSED`; its numbers are not the full results.

Without Docker: Python 3.13, `pip install -r requirements.txt` (or `conda env create -f environment.yml`), then `./reproduce.sh smoke` or `./reproduce.sh full`. The full run (all tasks, all analyses) takes many hours on a 10-core CPU; T5 and T6 need extra downloads (T6 about 300 MB).

Seeds are fixed. Results can still differ slightly across platforms and library versions. On macOS, scikit-learn boosting and PyTorch cannot share one process (segmentation fault), so `gbm_exp.py` runs in separate steps (`dump`, `fit`, `stats`).

After a full run, `python -I code/check_headlines.py` recomputes data counts, AUROC from the saved predictions, communication ratios and modelled decision impact from the raw files, without the project's metric code.

## Environment used
Apple M5 (10 cores, 16 GB), macOS, CPU only. Python 3.13, PyTorch 2.14, NumPy 2.3, pandas 2.3, scikit-learn 1.7, SciPy, statsmodels, Opacus 1.6.

## Licence and citation
MIT licence (`LICENSE`). Citation metadata are in `CITATION.cff`.
