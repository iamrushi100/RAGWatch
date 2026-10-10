# Controlled Retrieval Regression Experiment

Run the local experiment from the repository root:

```powershell
python -m src.regression_experiment
```

It evaluates `data/sample_dataset.json` with `k=2` by default and writes a
machine-readable result to
`reports/retrieval-regression-experiment.json`. To choose other inputs:

```powershell
python -m src.regression_experiment --dataset data/sample_dataset.json --k 2 --output reports/my-experiment.json
```

The experiment loads the dataset once. It evaluates the unchanged TF-IDF
baseline, then evaluates the same questions, chunks, and relevance labels
after reversing each complete baseline ranking. The reversal is the only
degraded setting. It is intentionally severe and exists to demonstrate that
the metrics can register a ranking regression; it is not proposed as a useful
retriever. The production retriever and dataset are not altered.

The JSON contains baseline and degraded per-question and aggregate Recall@k,
MRR, and nDCG@k values, plus each degraded aggregate score minus its baseline
score. A negative difference indicates a lower score in the degraded
condition. The numbers are calculated when the command runs and are not
hardcoded in the experiment implementation.

This is a controlled simulation on a tiny fictional fixture. It shows how
RAGWatch records a known ranking perturbation; it does not establish general
retrieval quality or prove detection of real-world drift. Reversing a complete
ranking is an artificial stress case, and results depend on the sample labels
and the selected `k`.
