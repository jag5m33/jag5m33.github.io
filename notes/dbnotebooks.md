# Notebooks: Python & SQL

A Databricks notebook is like a Jupyter notebook. The difference: **each cell can be Python or SQL**, and `spark` is already set up for you.

**On this page**

- Cells and magic commands
- Python cells
- SQL cells
- Running a notebook

## Cells and magic commands

> **Definition:** a **magic command** on the first line of a cell changes what the cell does.

| First line | The cell runs |
|---|---|
| (nothing) | Python |
| `%sql` | SQL |

## Python cells

```python
display(spark.table("elec_silver").limit(5))   # shows a table you can scroll and sort
df.printSchema()                                # every column and its type
print(df.count())                               # number of rows
```

- `display(...)` is the Databricks way to show a table. It's nicer than `print`.
- `spark` is ready to use. No setup needed.

## SQL cells

Saved tables can be queried straight away by name:

```sql
%sql
SELECT DATE(timestamp)              AS day,
       COUNT(*)                     AS flagged_half_hours,
       ROUND(MIN(anomaly_score), 3) AS worst_if_score
FROM elec_anomalies
WHERE is_anomaly
GROUP BY DATE(timestamp)
ORDER BY flagged_half_hours DESC;
```

This is how I checked the model in `04_analysis`: which days had the most flagged half-hours.

Two Databricks date functions I used:

| Function | Gives |
|---|---|
| `DATE(timestamp)` | the day, so half-hours can be grouped by day |
| `DATE_FORMAT(timestamp, 'EEEE')` | the day name, e.g. "Sunday" (`'MMMM'` gives the month name) |

> **Where it goes wrong:** a comma just before `FROM` is a syntax error. More SQL in [SQL](#/t/sqlbasics).

## Running a notebook

- **Run all**: runs every cell top to bottom. A notebook must pass Run all before it can go into a job.
- If a cell errors halfway, rerun from the cell that **loads** the data, so you start clean.

> **Where it goes wrong:** `IOStream.flush timed out` is only a warning. It appears when a cell (like MSTL) runs for a long time. Nothing failed.

## My 4 notebooks

| Notebook | Does | In the job? |
|---|---|---|
| `01_load` | read the 6 CSVs, stack them, save `elec_stacked` | yes |
| `02_clean` | dates, timestamp, checks, calendar columns, save `elec_silver` | yes |
| `03_model` | MSTL, isolation forest + MLflow, save `elec_anomalies` | yes |
| `04_analysis` | SQL on the results | no: it's for looking |

## Quick reference

| Term | In one line |
|---|---|
| `%sql` | this cell is SQL |
| `display()` | show a table |
| Run all | every cell, top to bottom |
