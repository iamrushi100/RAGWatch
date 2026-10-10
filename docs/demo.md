# RAGWatch Local Demo

This walkthrough runs the existing sample evaluation, saves two history runs,
opens the read-only dashboard, and runs the controlled ranking-inversion
experiment. Run the commands from the repository root in Windows PowerShell
after following the [README setup instructions](../README.md#setup-on-windows-powershell).
No API keys or external services are needed.

The commands create two JSON files under `reports/` and one local SQLite
database under `data/`. Pick new JSON filenames if either report already
exists: the regular evaluation exporter can replace a file, while the
regression experiment refuses an existing output path. The SQLite file name
below ends in `.db`, which is ignored by the repository's `.gitignore`.

## 1. Evaluate and export a report

```powershell
python -m src.cli data/sample_dataset.json --k 2 --output reports/demo-evaluation.json --failure-analysis --min-mrr 0.5
```

The command prints one line of metrics per question and the aggregate
Recall@2, MRR, and nDCG@2 scores. It writes the complete JSON result to
`reports/demo-evaluation.json`. Because failure analysis and an MRR threshold
were requested, the report includes their diagnostic and threshold fields.
The measured threshold result depends on the scores from this run.

## 2. Save and compare two SQLite runs

Run the same evaluation twice with the same database path:

```powershell
python -m src.cli data/sample_dataset.json --k 2 --history-db data/ragwatch_demo.db
python -m src.cli data/sample_dataset.json --k 2 --history-db data/ragwatch_demo.db
```

Each successful command prints the saved run ID. The CLI does not have a
separate compare command. The dashboard uses the existing history API to list
these runs and compare a selected baseline and candidate. Comparisons require
matching dataset IDs and `k`; reported differences are candidate minus
baseline. Repeating an unchanged deterministic evaluation is useful to inspect
the flow, though it is not an experiment showing a retrieval change.

## 3. Inspect reports, diagnostics, and history in the dashboard

```powershell
streamlit run app.py
```

In the browser, choose `demo-evaluation.json` in the JSON report selector to
see the aggregate and per-question metrics, threshold status, alerts if any,
and failure-analysis labels. In the SQLite history section, choose
`ragwatch_demo.db`, then select baseline and candidate run IDs to view their
aggregate and per-question comparisons. The dashboard reads existing files; it
does not rerun the evaluation when the page refreshes.

## 4. Run and inspect the controlled regression experiment

Choose an unused report filename, then run:

```powershell
python -m src.regression_experiment --dataset data/sample_dataset.json --k 2 --output reports/demo-regression-experiment.json
```

The experiment evaluates the normal TF-IDF ranking, then reverses each
complete ranking and calculates the same metrics again. It writes measured
baseline and degraded results, plus candidate-minus-baseline differences.
Inspect this experiment-specific JSON in the terminal with Python's standard
JSON formatter:

```powershell
python -m json.tool reports/demo-regression-experiment.json
```

The dashboard may list this JSON file because it is under `reports/`, but the
regular-report loader expects a different structure and will show a validation
message if you select it. Inspect the experiment report with `json.tool`
instead. The inversion is an artificial stress test that shows how metrics
respond to a deliberately reversed ranking. It is not realistic production
drift and does not show that RAGWatch detects real-world drift generally. The
fixture is small and fictional, so these results do not establish general
retrieval quality. See [`retrieval_regression.md`](retrieval_regression.md) for more
detail.
