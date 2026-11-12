#!/bin/sh
# One-command reproduction.   ./reproduce.sh smoke   -> 2 seeds / 20 rounds on T1, writes to ./results_smoke (minutes)
#                              ./reproduce.sh full    -> all experiments (hours; about 4-5 h on a 10-core CPU)
set -e
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mode="${1:-smoke}"
[ -f data/cdc_diabetes.csv ] || sh download_data.sh
[ -d data/nhanes ] && [ -f data/nhanes_processed.csv ] || { python download_nhanes.py; (cd code && python nhanes_prep.py); }
if [ "$mode" = "smoke" ]; then
  export REPRO_SMOKE=1 HIPERC_RESULTS="$(pwd)/results_smoke"; mkdir -p "$HIPERC_RESULTS"
  cd code
  python tune.py T1 && python run_main.py T1 main && python clinical.py T1 && python analyze.py T1 > "$HIPERC_RESULTS/analyze_T1.txt" 2>&1
  python -c "import pandas as pd; d=pd.read_csv('$HIPERC_RESULTS/runs_T1_main.csv'); print(d.groupby('spec').auroc.mean().round(3).sort_values().to_string()); assert len(d)>0; print('SMOKE TEST PASSED')"
  python validity_checks.py > /dev/null 2>&1 && echo "validity checks ran"
  python leakage_audit.py T1 > "$HIPERC_RESULTS/leakage_audit_T1.txt" 2>&1 && echo "leakage audit ran"
  GBM_TASKS=T1 python gbm_exp.py dump > /dev/null 2>&1 && GBM_TASKS=T1 python gbm_exp.py fit > "$HIPERC_RESULTS/gbm_T1.txt" 2>&1 && echo "boosted-tree reference ran"
  exit 0
fi
cd code
for t in T1 T2 T3 T4; do python tune.py $t; python run_main.py $t main; python clinical.py $t; done
python loho.py; python drift.py; python run_main.py T2 skew; python dp_exp.py; python importance.py; python linear_exp.py; python loho_cdc.py
python nhanes_exp.py; python nhanes_drift.py; python nhanes_drift.py samelr; python ft_exp.py; python lr_sensitivity.py; python lr_sensitivity_low.py; python init_sensitivity.py; python validity_checks.py
# FLamby cross-check (needs a clone of https://github.com/owkin/FLamby at /tmp/flamby_try/FLamby)
[ -d /tmp/flamby_try/FLamby ] || { mkdir -p /tmp/flamby_try; git clone --depth 1 https://github.com/owkin/FLamby.git /tmp/flamby_try/FLamby; }
python crosscheck_flamby.py; python crosscheck_flamby.py same_init
for t in T1 T2 T3 T4; do python analyze.py $t; python stats_patient.py $t first; done
for t in T1 T2 T3; do python stats_patient.py $t first lin_hiperc; done
# T5: open sepsis data from two hospital systems (download ~160 MB, about 1 hour; cohort, tuning, all methods, cross-system validation, statistics)
[ -f ../data/sepsis_processed.csv ] || { (cd .. && python -I download_sepsis.py) && python -I sepsis_prep.py; }
python tune.py T5; python run_main.py T5 main; python clinical.py T5; python ft_exp.py T5; python loho_sepsis.py; python stats_patient.py T5 first; python ablation_patient.py T5
# T5 extras: fixed-horizon outcome (T5H24), federated linear models, recalibration
python -I ../code/sepsis_prep.py 2>/dev/null || (cd .. && python -I code/sepsis_prep.py)
cp ../results/configs_T5.json ../results/configs_T5H24.json 2>/dev/null; python tune.py T5H24; python run_main.py T5H24 main; python clinical.py T5H24; python stats_patient.py T5H24 first; python loho_sepsis.py T5H24
python linear_sepsis.py; python stats_patient.py T5 first; python stats_patient.py T5 first lin_hiperc; python recal_sepsis.py
# T6: Mexican COVID-19 registry (download about 300 MB), forward in time; impact analysis
[ -f ../data/mx_processed.csv.gz ] || { (cd .. && python -I download_mx.py && python -I code/mx_prep.py); }
python mx_leakage_check.py; python tune.py T6; python run_main.py T6 main; python clinical.py T6; python ft_exp.py T6; python linear_mx.py; python loho_mx.py
python stats_patient.py T6 first; python stats_patient.py T6 first lin_hiperc; python ablation_patient.py T6; python recal_mx.py; python impact_analysis.py T2 T4 T5 T6
# audit for hidden flaws (arrays go to data/audit_tmp), independent re-implementation, and the T6 frozen-quarter and rolling-origin replay
python leakage_audit.py; python -I audit_independent.py; python rolling_mx.py
# pooled gradient-boosted trees (three processes: torch dump, scikit-learn fit, torch statistics) and heterogeneity diagnostics
python gbm_exp.py dump; python gbm_exp.py fit; python gbm_exp.py dumpx; python gbm_exp.py fitx; python gbm_exp.py stats; python heterogeneity_rule.py
# federated tree ensemble, recalibration policies in the T6 replay, and the one-command check of headline numbers
python fedgbm.py equivalence; python fedgbm.py run; python fedgbm.py stats; python recal_policy_mx.py; python -I check_headlines.py
echo "Done. Results are in results/."
