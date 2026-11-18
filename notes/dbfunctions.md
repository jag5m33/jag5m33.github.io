# pyspark.sql functions & windows

`pyspark.sql.functions` is PySpark's toolbox for columns: dates, maths, if/else. A **window** lets you number or rank rows in an order.

**On this page**

- Importing the toolbox
- Dates and times
- If/else and missing values
- Windows: numbering rows in order
- How I built the timestamp

## Importing the toolbox

```python
from pyspark.sql import functions as F
from pyspark.sql import Window
```

`F.` is used a lot, a bit like `np.` in numpy.

## Dates and times

```python
spark.conf.set("spark.sql.session.timeZone", "UTC")   # first: stop Spark shifting times for summer time
```

| Function | Does | Example |
|---|---|---|
| `F.to_date(col, "dd-MMM-yyyy")` | text → date | `'01-JAN-2021'` → 2021-01-01 |
| `F.year` / `F.month` | year / month number | 2018, 1 |
| `F.dayofweek` | day number (Sunday = 1) | |
| `F.hour` / `F.minute` | hour / minute | |
| `F.from_utc_timestamp(col, "Europe/London")` | GMT → UK clock time | |
| `F.timestamp_seconds` | seconds since 1970 → a date and time | |

The calendar columns I added in `02_clean`:

```python
local = F.from_utc_timestamp("timestamp", "Europe/London")    # UK clock time

df = (df
      .withColumn("year", F.year("timestamp"))
      .withColumn("month", F.month("timestamp"))
      .withColumn("weekday", (F.dayofweek(local) + 5) % 7)     # Monday = 0 ... Sunday = 6
      .withColumn("half_hour_intervals", F.hour(local) * 2 + F.floor(F.minute(local) / 30)))   # 0-47
```

`dayofweek` counts Sunday as 1, so `+ 5) % 7` shifts it to Monday = 0, to match my pandas version.

## If/else and missing values

| Function | Does |
|---|---|
| `F.col("x")` | refer to a column |
| `F.lit(5)` | a plain number Spark can use in a calculation |
| `F.when(condition, value)` | if/else |
| `F.col("x").isNull()` | is it blank? |
| `F.coalesce(a, b)` | first one that isn't blank |

Some years wrote dates as `01-JAN-2021`, others as `01-Jan-23`. So: try one format, and if it fails, the other.

```python
df = df.withColumn("settlement_date", F.coalesce(
    F.expr("try_to_date(settlement_date, 'dd-MMM-yyyy')"),   # 4-digit year
    F.expr("try_to_date(settlement_date, 'dd-MMM-yy')")))    # 2-digit year

df.select(F.min("settlement_date"), F.max("settlement_date"),
          F.count(F.when(F.col("settlement_date").isNull(), 1)).alias("bad_dates")).show()
# 2018-01-01, 2023-12-31, 0
```

- `try_to_date` gives a blank instead of crashing when the format doesn't fit.
- `F.expr("...")` runs a bit of SQL inside PySpark.

## Windows: numbering rows in order

> **Definition:** a **Window** sets an order to go through the rows in. `F.row_number().over(w)` then numbers the rows 1, 2, 3 … in that order.

Spark data has no fixed order, so you spell out the order every time you need one.

## How I built the timestamp

Same idea as `pd.date_range`: put the rows in time order, then count up from 2018-01-01 00:00 GMT in 30-minute steps.

```python
w = Window.orderBy("settlement_date", "settlement_period")     # by date, then period 1-48

df = df.withColumn("row_num", F.row_number().over(w))          # number the rows 1, 2, 3 ...

start = 1514764800                                              # 2018-01-01 00:00 GMT, in seconds since 1970
df = df.withColumn("timestamp",
                   F.timestamp_seconds(F.lit(start) + (F.col("row_num") - 1) * 1800))   # + 30 min per row
df = df.drop("row_num")
```

Check: min 2018-01-01 00:00, max 2023-12-31 23:30, and 105,168 different timestamps (one per row).

## Quick reference

| Term | In one line |
|---|---|
| `F` | the PySpark column toolbox |
| `F.to_date` | text → date |
| `F.when` | if/else |
| `F.coalesce` | first non-blank value |
| `Window.orderBy(...)` | the order to number rows in |
| `F.row_number().over(w)` | number the rows 1, 2, 3 … |
