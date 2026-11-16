# GROUP BY & HAVING

Collapse rows into groups and calculate one number per group.

**On this page**

- Aggregate functions
- GROUP BY
- HAVING vs WHERE
- The one check before every GROUP BY query
- Grouping by month

## Aggregate functions

> **Definition:** an **aggregate** turns many rows into one value: `COUNT`, `SUM`, `AVG`, `MIN`, `MAX`.

```sql
SELECT COUNT(*) FROM Videos;                       -- all rows
SELECT COUNT(DISTINCT user_id) FROM Interactions;  -- unique users
```

## GROUP BY

> **Definition:** `GROUP BY` collapses rows that share a value into one group, so aggregates are calculated **per group**.

```sql
SELECT sender_id, COUNT(message_id) AS message_count
FROM messages
WHERE EXTRACT(MONTH FROM sent_date) = 8 AND EXTRACT(YEAR FROM sent_date) = 2022
GROUP BY sender_id
ORDER BY message_count DESC
LIMIT 2;                     -- the 2 people who sent the most messages in Aug 2022
```

- Usually group by a **key** (an id), as these are unique.

## HAVING vs WHERE

> **Definition:** `WHERE` drops **rows** before grouping. `HAVING` drops **groups** after grouping, so it can use aggregates.

```sql
SELECT customer_id, SUM(total_amount) AS total_spent
FROM orders
GROUP BY customer_id
HAVING SUM(total_amount) > 10000;     -- only customers who spent over 10,000
```

## The one check

Before every query, run the check:

> **Say it like this:** "Every column in SELECT is either in the GROUP BY or inside an aggregate."

That's my one persistent bug. Ten seconds each time.

## Grouping by month

```sql
SELECT
    STRFTIME('%m', InvoiceDate) AS invoice_month,   -- extracts the month (01-12)
    COUNT(InvoiceId) AS total_invoices
FROM invoices
WHERE InvoiceDate LIKE '2010%'                      -- filters for the year 2010
GROUP BY STRFTIME('%m', InvoiceDate)                -- groups the data by month
HAVING COUNT(InvoiceId) > 20                        -- months with more than 20 invoices
ORDER BY invoice_month ASC;
```

Date functions differ by database:

| Database | Month from a date |
|---|---|
| SQLite | `STRFTIME('%m', date)` |
| PostgreSQL | `EXTRACT(MONTH FROM date)` |
| MySQL | `MONTH(date)`, and `DATEDIFF(a, b)` for days between |

```sql
-- MySQL: days between each user's first and last post in 2021
SELECT user_id,
       DATEDIFF(MAX(DATE(post_date)), MIN(DATE(post_date))) AS days_difference
FROM posts
WHERE YEAR(post_date) = 2021
GROUP BY user_id
HAVING days_difference > 0;
```

> **Check:** `HAVING days_difference > 0` uses a SELECT alias inside HAVING. MySQL allows it; PostgreSQL and standard SQL don't, so there you'd repeat the expression.

## Quick reference

| Term | In one line |
|---|---|
| Aggregate | many rows → one value (COUNT, SUM, AVG, MIN, MAX) |
| `GROUP BY` | one output row per group |
| `WHERE` | filters rows, before grouping |
| `HAVING` | filters groups, after grouping |
| The check | every SELECT column is grouped or aggregated |
