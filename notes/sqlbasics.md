# SELECT, DISTINCT & LIMIT

The shape of every query, and the order SQL actually runs it in.

**On this page**

- SELECT … FROM
- DISTINCT: remove duplicates
- AS: rename columns
- LIMIT and OFFSET
- The order SQL runs a query

## SELECT … FROM

> **Definition:** `SELECT` picks the columns you want; `FROM` says which table they come from.

```sql
SELECT * FROM Videos;                       -- * = select all columns from the table
SELECT title, views FROM Videos;            -- just these columns
SELECT title AS video_title FROM Videos;    -- rename the column to a more readable name
```

## DISTINCT

> **Definition:** `DISTINCT` removes duplicate values from the result.

```sql
SELECT DISTINCT column_name FROM table_name;
SELECT DISTINCT col_name_1, col_name_2 FROM table_name;   -- unique combinations of both

SELECT DISTINCT user_id, interaction_type FROM Interactions ORDER BY user_id;

-- how many unique users are currently active
SELECT COUNT(DISTINCT user_id) AS active_engagers FROM Interactions;
```

- Missing values: DISTINCT handles NULL the same way as other repeats: it de-duplicates them and returns **one** NULL.

> **Check:** in my notes the second query is labelled "get a list of unique emails", but it returns unique **(user_id, interaction_type)** pairs. Unique emails would be `SELECT DISTINCT email FROM Users;`.

**DISTINCT vs GROUP BY**

| | Use it when |
|---|---|
| `DISTINCT` | simple syntax: you just want duplicates removed |
| `GROUP BY` | you need a calculation per group: COUNT, SUM, AVG etc. ([GROUP BY & HAVING](#/t/sqlgroup)) |

## LIMIT and OFFSET

> **Definition:** `LIMIT` restricts how many rows come back. `OFFSET` skips some rows first.

```sql
SELECT username, email, join_date FROM Users LIMIT 5;     -- the first 5 users

-- skip the first 5 videos and return the next 5
SELECT video_id, title, views FROM Videos LIMIT 5 OFFSET 5;
```

- OFFSET is like skipping a few pages of a book before you start reading: it's how **pagination** works (page 2 at 4 per page = `LIMIT 4 OFFSET 4`, items 5 to 8).
- SQLite needs the `OFFSET` keyword; it doesn't support MySQL's comma form (`LIMIT 10, 5`).
- Always use LIMIT when exploring large tables: loading 10 million rows at once is slow.

> **Where it goes wrong:** without `ORDER BY`, the database returns rows in an **arbitrary** order, so LIMIT is unpredictable. Always pair LIMIT with ORDER BY.

```sql
-- the top 3 most-viewed videos
SELECT video_id, title, views
FROM Videos
ORDER BY views DESC
LIMIT 3;
```

## The order SQL runs a query

You **write** `SELECT` first, but SQL **runs** it fifth:

| Step | Clause | What it does |
|---|---|---|
| 1 | `FROM` / `JOIN` | build the source table |
| 2 | `WHERE` | drop rows |
| 3 | `GROUP BY` | collapse rows into groups |
| 4 | `HAVING` | drop groups |
| 5 | `SELECT` | pick and compute columns (aliases are born here), then `DISTINCT` |
| 6 | `ORDER BY` | sort |
| 7 | `LIMIT` | cut |

> **Check:** in my notes I wrote "DISTINCT after SELECT, then followed by ORDER BY / GROUP BY". GROUP BY actually runs **before** SELECT (step 3). DISTINCT runs right after SELECT, then ORDER BY, then LIMIT.

This order explains two rules you'll meet later:

- You **can't** use a SELECT alias in WHERE (WHERE runs before the alias exists). See [Aliases](#/t/sqlalias).
- You **can** use a SELECT alias in ORDER BY (it runs after).

## Quick reference

| Term | In one line |
|---|---|
| `SELECT … FROM` | which columns, from which table |
| `*` | all columns |
| `DISTINCT` | remove duplicate rows (one NULL kept) |
| `LIMIT n OFFSET k` | skip k rows, return the next n |
| Run order | FROM → WHERE → GROUP BY → HAVING → SELECT → ORDER BY → LIMIT |
