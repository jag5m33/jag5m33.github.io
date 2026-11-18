# The Databricks workspace

Databricks is one place where your **notebooks, data and compute** all live together. Your code runs on Databricks' machines, not your laptop.

**On this page**

- The left menu: what each part is for
- Git folder: bringing in the GitHub repo
- Catalog: volumes and tables
- Compute: what runs the code
- Experiments: where MLflow runs go

## The left menu

| Menu item | What I used it for |
|---|---|
| **Workspace** | my files: the Git folder with the notebooks and `src/` |
| **Catalog** | my data: the raw CSVs (in a volume) and the saved tables |
| **Jobs & Pipelines** | running the 3 notebooks in order as one pipeline ([Jobs & pipelines](#/t/dbjobs)) |
| **Compute** | what runs the code: I used **Serverless** |
| **Experiments** | my MLflow runs |

## Git folder

> **Definition:** a **Git folder** is a copy of a GitHub repo inside Databricks.

- Workspace → Create → **Git folder** → paste the repo URL.
- My notebooks and `src/` appear in Databricks, so `from src.mstl import ...` still works.
- The Databricks notebooks went in a new `databricks/` folder in the repo.

## Catalog: volumes and tables

> **Definition:** a **volume** is a folder for **files** (like my raw CSVs). A **table** is saved **data** you can read by name, from Python or SQL.

Every name has 3 parts: **catalog . schema . name**, e.g. `workspace.default.elec_raw`.

| | What | Example |
|---|---|---|
| Volume | files | `/Volumes/workspace/default/elec_raw/` holds the 6 `demanddata_20XX.csv` files |
| Table | data | `elec_stacked`, `elec_silver`, `elec_mstl`, `elec_anomalies` |

- Make a volume: Catalog → workspace → default → Create → **Volume**, then Upload the files.
- `../data/...` paths from my laptop become `/Volumes/...` paths here.

> **Where it goes wrong:** saving to a volume that doesn't exist gives `Cannot save file into a non-existent directory`. Create the volume first.

### My tables, one per stage

| Table | Made by | Holds |
|---|---|---|
| `elec_stacked` | `01_load` | the 6 years stacked into one table |
| `elec_silver` | `02_clean` | cleaned data + timestamp + calendar columns |
| `elec_mstl` | `03_model` | MSTL output: trend, seasonals, residual |
| `elec_anomalies` | `03_model` | every half-hour with its anomaly score and flag |

Each stage saves a table, and the next stage reads it.

## Compute

> **Definition:** **compute** is the machine that runs your code.

- I used **Serverless**: Databricks gives you a machine when you run something. Nothing to set up or switch off.
- The **Compute** tab lists the compute in the workspace. Notebooks and the job both ran on Serverless.

## Experiments: MLflow runs

- MLflow is built in. Every `mlflow.start_run(...)` in a notebook is saved to that **notebook's experiment**.
- Open it from the **flask icon** in the notebook's right sidebar, or the **Experiments** page.
- I named runs by version (`isolation_forest`, then `isolation_forest_v2`) so I could compare them.

> **Where it goes wrong:** delete `mlflow.set_tracking_uri(...)` and `mlflow.set_experiment(...)` from the laptop code. Databricks tracks the runs itself.

## Quick reference

| Term | In one line |
|---|---|
| Workspace | where the files and notebooks live |
| Git folder | the GitHub repo inside Databricks |
| Volume | a folder for files: `/Volumes/catalog/schema/volume/` |
| Table | saved data, read by name |
| Serverless | compute Databricks starts for you |
| Experiment | where MLflow saves a notebook's runs |
