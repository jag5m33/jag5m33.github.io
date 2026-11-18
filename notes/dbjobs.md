# Jobs & pipelines

A **job** runs my notebooks in order, with one click, with nobody watching.

**On this page**

- What a job is
- Building my job
- Running it
- Before a notebook goes in a job

## What a job is

> **Definition:** a **job** is a set of **tasks**. Each task runs one notebook. A task can **depend on** another, meaning "don't start until that one has finished".

The chain of tasks is called a **DAG**: it only goes one way, with no loops. That's how data pipelines are run.

```
load  →  clean  →  model
01_load   02_clean   03_model
```

## Building my job

1. Left menu → **Jobs & Pipelines** → **Create** → **Job**.
2. Name it: `elec_anomaly_pipeline`.
3. Task 1: name `load`, type **Notebook**, source **Workspace**, path `databricks/01_load`, compute **Serverless** → Create task.
4. **+ Add task** → Notebook: name `clean`, path `02_clean`, **Depends on: `load`**.
5. Same again: `model`, path `03_model`, **depends on `clean`**.

## Running it

- Click **Run now**, then open the run from the **Runs** tab.
- Each task is a box. It turns **green** when it finishes.
- If a box goes **red**, the tasks after it don't run. Click the box to see the notebook output and the error.
- No schedule: my data doesn't change, so one run shows it works.

## Before a notebook goes in a job

> **Where it goes wrong:** a job runs every cell, top to bottom, and stops at the first error.

- Each notebook must pass **Run all** on its own first.
- Delete cells that only worked by hand: test prints, saves to a volume that doesn't exist, cells that reload a table to skip a step.
- `04_analysis` stays out of the job: it's for looking at the results.

## Quick reference

| Term | In one line |
|---|---|
| Job | notebooks run as one pipeline |
| Task | one notebook in the job |
| Depends on | wait for that task to finish first |
| DAG | the one-way chain of tasks |
| Run now / Runs tab | start a run / see how it went |
