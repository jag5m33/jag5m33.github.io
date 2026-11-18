# PySpark basics

PySpark is Python for **Spark**. Spark splits data across machines so it can handle data too big for one computer. The code looks a lot like pandas.

**On this page**

- PySpark vs pandas
- Reading and stacking files
- Changing columns
- Checking the data
- Saving and loading tables
- Moving between Spark and pandas

## PySpark vs pandas

> **Definition:** a **Spark DataFrame** is a table spread across machines. It has **no row order** and **no index**.

| pandas | PySpark |
|---|---|
| `pd.read_csv(path)` | `spark.read.csv(path, header=True, inferSchema=True)` |
| `len(df)` | `df.count()` |
| `df[["a", "b"]]` | `df.select("a", "b")` |
| `df["new"] = ...` | `df = df.withColumn("new", ...)` |
| `df.drop(columns=[...])` | `df.drop("a", "b")` |
| `df[df["x"] == 0]` | `df.filter(F.col("x") == 0)` |
| `df.groupby("a").size()` | `df.groupBy("a").count()` |
| `df.head()` | `display(df.limit(5))` |

## Reading and stacking files

```python
path = "/Volumes/workspace/default/elec_raw/"

dfs = [spark.read.csv(f"{path}demanddata_{year}.csv",
                      header=True, inferSchema=True)
       for year in range(2018, 2024)]

# stack the years, matching columns by NAME
df = dfs[0]
for d in dfs[1:]:
    df = df.unionByName(d, allowMissingColumns=True)

print(df.count())    # 105168
```

- `header=True`: the first row is the column names.
- `inferSchema=True`: work out the types (numbers, text) instead of reading everything as text.
- `unionByName(..., allowMissingColumns=True)` stacks tables **by column name**, and fills a missing column with nulls. In pandas this took a column-matching loop.

> **Where it goes wrong:** reading the whole folder at once lines columns up by **position**. My years have different columns, so values would land under the wrong headings with no error.

## Changing columns

```python
df = df.toDF(*[c.lower() for c in df.columns])                    # lowercase every column name
df = df.drop("embedded_wind_capacity", "embedded_solar_capacity")  # remove columns
df = df.withColumn("year", F.year("timestamp"))                     # add a column
```

- `withColumn("name", ...)` makes a new column, or replaces it if the name exists.

> **Where it goes wrong:** a typo in a column name gives `UNRESOLVED_COLUMN ... Did you mean ...`. Spark lists the real names. A typo in the **new** name of `withColumn` gives no error: it quietly makes a new, misspelled column.

## Checking the data

My 4 checks in `02_clean`, all in PySpark:

| Check | Result |
|---|---|
| dates without 48 half-hours | 12 (the clock-change days) |
| duplicate half-hours | 0 |
| nulls in the flow columns | 87,648 in `scottish_transfer`, 17,520 in the other four (columns that didn't exist yet) |
| `england_wales_demand` = 0 | 3 rows in 2018, left as they are |

```python
display(df.groupBy("settlement_date").count().filter(F.col("count") != 48))   # dates without 48 rows
display(df.filter(F.col("england_wales_demand") == 0))                         # the 3 zero rows
```

## Saving and loading tables

```python
df.write.mode("overwrite").saveAsTable("elec_stacked")   # save as a table
df = spark.table("elec_stacked")                          # load it again, by name
```

- `mode("overwrite")` replaces the table, so the cell is safe to run again.
- Tables are saved in **Delta** format (parquet plus a history log).

> **Where it goes wrong:** adding new columns to a table that already exists fails, even with overwrite. Add `.option("overwriteSchema", "true")` to allow it:
>
> `df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("elec_silver")`

## Moving between Spark and pandas

MSTL and isolation forest are pandas / scikit-learn code, so `03_model` moves the data into pandas and back.

```python
# Spark → pandas
pdf = spark.table("elec_silver").select(cols).orderBy("timestamp").toPandas()
pdf = pdf.set_index("timestamp")          # assign it back, or nothing changes

# pandas → Spark table
model_df.attrs = {}
spark.createDataFrame(model_df.reset_index()).write.mode("overwrite") \
     .option("overwriteSchema", "true").saveAsTable("elec_anomalies")
```

- **`orderBy` before `toPandas`**: Spark keeps no row order, and MSTL needs time order.
- **`reset_index()`** before saving: Spark tables have no index, so the timestamp has to be a column.

> **Where it goes wrong:** `TypeError: Object of type PlanMetrics is not JSON serializable`. `toPandas()` attaches hidden Spark info to the DataFrame. Run `df.attrs = {}` before `to_parquet` or `spark.createDataFrame`.

## Quick reference

| Command | Does |
|---|---|
| `spark.read.csv(...)` | read a CSV |
| `unionByName(..., allowMissingColumns=True)` | stack tables by column name |
| `withColumn` / `drop` / `select` / `filter` | change, remove, pick, keep rows |
| `groupBy(...).count()` | rows per group |
| `saveAsTable` / `spark.table` | save / load a table |
| `toPandas()` / `createDataFrame()` | Spark → pandas / pandas → Spark |
