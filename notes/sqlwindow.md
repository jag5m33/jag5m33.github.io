# Window functions

Calculate across related rows **without** collapsing them into groups.

**On this page**

- OVER: the window
- PARTITION BY and ORDER BY
- Ranking, running totals, previous row

> **Definition:** the `OVER` clause defines a "window" of rows for a window function (like `ROW_NUMBER()`, `SUM()` or `AVG()`) to look at and calculate a value for.

The rest of this page is **added, not from my notes**: a short starter.

## GROUP BY vs a window

| | Rows out |
|---|---|
| `GROUP BY user_id` | one row per user |
| `SUM(views) OVER (PARTITION BY user_id)` | every row kept, each with its user's total next to it |

## PARTITION BY and ORDER BY

- `PARTITION BY col`: split the rows into groups (like GROUP BY, but rows stay).
- `ORDER BY col`: the order inside each group (needed for rankings and running totals).

```sql
SELECT user_id, video_id, views,
       SUM(views) OVER (PARTITION BY user_id)                       AS user_total,
       ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY views DESC) AS rank_in_user
FROM Videos;
```

## Ranking, running totals, previous row

| Function | Gives |
|---|---|
| `ROW_NUMBER()` | 1, 2, 3, 4: always unique |
| `RANK()` | 1, 2, 2, 4: ties share, then a gap |
| `DENSE_RANK()` | 1, 2, 2, 3: ties share, no gap |
| `SUM(x) OVER (ORDER BY date)` | running total |
| `LAG(x)` / `LEAD(x)` | the previous / next row's value |

```sql
-- each user's top video only
WITH ranked AS (
    SELECT user_id, title, views,
           ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY views DESC) AS rn
    FROM Videos
)
SELECT user_id, title, views FROM ranked WHERE rn = 1;
```

> **Where it goes wrong:** you can't filter on a window function in WHERE (it runs at the SELECT step). Wrap it in a CTE, then filter, as above.

## Quick reference

| Term | In one line |
|---|---|
| `OVER (…)` | turns an aggregate into a window function |
| `PARTITION BY` | groups, but rows are kept |
| `ROW_NUMBER` / `RANK` / `DENSE_RANK` | numbering within a group |
| `LAG` / `LEAD` | look one row back / ahead |
