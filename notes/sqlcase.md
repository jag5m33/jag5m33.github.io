# CASE WHEN & FILTER

If/then logic inside a query: label rows, and count only the rows you want.

**On this page**

- CASE WHEN
- CASE inside an aggregate (conditional counts)
- FILTER (WHERE …)
- COALESCE and NULLIF

## CASE WHEN

> **Definition:** `CASE WHEN` returns a value depending on a condition, like an if/else, and creates a new column.

```sql
SELECT device_type,
       CASE WHEN prices > 100 THEN 1 ELSE 0 END AS upper_range,
       CASE WHEN prices < 100 THEN 1 ELSE 0 END AS lower_range
FROM pricing;
```

> **Check:** in my notes there's a stray `ELSE not_priced` line after the two CASEs. An `ELSE` only lives **inside** a `CASE … END`, so that line is a syntax error. Note too that a price of exactly 100 lands in neither range.

Several conditions are checked **top to bottom**, and the first TRUE one wins:

```sql
SELECT title, views,
       CASE
           WHEN views >= 10000 THEN 'viral'
           WHEN views >= 1000  THEN 'popular'
           ELSE 'small'          -- everything else, incl. NULL
       END AS reach
FROM Videos;
```

## CASE inside an aggregate (added, not from my notes)

Put CASE **inside** SUM or COUNT to count only some rows. This is called **conditional aggregation**, and it turns rows into columns:

```sql
SELECT video_id,
       SUM(CASE WHEN interaction_type = 'like'    THEN 1 ELSE 0 END) AS likes,
       SUM(CASE WHEN interaction_type = 'comment' THEN 1 ELSE 0 END) AS comments
FROM Interactions
GROUP BY video_id;
```

## FILTER (WHERE …) (added, not from my notes)

> **Definition:** `FILTER (WHERE …)` after an aggregate makes it look only at the rows that pass, a shorter way to write the CASE version.

```sql
SELECT video_id,
       COUNT(*) FILTER (WHERE interaction_type = 'like')    AS likes,
       COUNT(*) FILTER (WHERE interaction_type = 'comment') AS comments
FROM Interactions
GROUP BY video_id;
```

| | Works in |
|---|---|
| `SUM(CASE WHEN …)` | every database |
| `FILTER (WHERE …)` | PostgreSQL and SQLite (3.30+); **not** MySQL or SQL Server |

> **Where it goes wrong:** WHERE drops rows for **every** column; FILTER only affects the one aggregate it's attached to. Use FILTER (or CASE) when different columns need different conditions in one pass.

## COALESCE and NULLIF (added, not from my notes)

```sql
SELECT COALESCE(comment_text, '(no comment)') FROM Interactions;   -- first non-NULL value

SELECT likes / NULLIF(views, 0) AS like_rate FROM video_stats;     -- NULL instead of divide-by-zero
```

## Quick reference

| Term | In one line |
|---|---|
| `CASE WHEN … THEN … ELSE … END` | if/else that makes a new column |
| `SUM(CASE WHEN … THEN 1 ELSE 0 END)` | count only matching rows |
| `COUNT(*) FILTER (WHERE …)` | the same, shorter (Postgres, SQLite) |
| `COALESCE(a, b)` | first value that isn't NULL |
| `NULLIF(a, 0)` | turn 0 into NULL (safe division) |
