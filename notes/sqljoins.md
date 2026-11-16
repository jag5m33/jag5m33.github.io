# Joins

A join glues two tables side by side, matching rows on a shared value.

**On this page**

- What a join does
- The join types
- LEFT JOIN to find what's missing
- Self join
- The two rules for joins

## What a join does

> **Definition:** a **join** connects data from different tables so you can view it together in one result.

Think of tables as separate spreadsheets: one lists **customers** (name, id, email), another lists **orders** (order_id, date, product, customer_id). The orders sheet records **who** bought each item by `customer_id`, not by name. A JOIN tells SQL: find the matching customer_id in both sheets and stitch the rows together.

```sql
SELECT c.customer_name, o.order_date
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id;
```

## The join types

| Type | Keeps | Use when |
|---|---|---|
| `INNER JOIN` (= `JOIN`) | only matching rows | the default: you're decorating rows with info from elsewhere |
| `LEFT JOIN` | all left rows, NULLs on the right | the left table is the subject: "every", "all", "including", "never" |
| `RIGHT JOIN` | all right rows | never: swap the tables and use LEFT |
| `FULL OUTER JOIN` | unmatched rows from both sides | rare: both tables have orphans you care about |
| `SELF JOIN` | n/a | a table joined to itself: not a keyword, a use, with two aliases |
| `CROSS JOIN` | every pairing | generating combinations; no ON clause |
| `UNION` | n/a | not a join: stacks results **vertically**. `UNION ALL` keeps duplicates |

## LEFT JOIN to find what's missing

> **Definition:** `LEFT JOIN` keeps **every row of the left table**. Anything with no match in the right table gets **NULL** in the right table's columns.

```sql
-- artists with no albums
SELECT ar.Name
FROM artists ar
LEFT JOIN albums al ON ar.ArtistId = al.ArtistId
WHERE al.AlbumId IS NULL;    -- no match on the right = NULL
```

> **Check:** in my notes I wrote "left = left of the equals sign in the join". The left table is the one written **before** `LEFT JOIN` (in FROM), here `artists`. The order inside `ON a = b` doesn't matter.

## Self join

A table joined to itself, e.g. to find each employee's manager:

```sql
SELECT
    e1.first_name AS employee_name,
    e2.first_name AS manager_name
FROM employees AS e1                       -- e1 = the workers
LEFT JOIN employees AS e2                  -- e2 = a second copy, treated as the managers
  ON e1.manager_id = e2.employee_id;       -- worker's manager_id → find that employee_id in e2
```

- **Two nicknames, one table:** e1 and e2 let SQL treat the same table as two separate spreadsheets.
- **Why LEFT JOIN:** top bosses have no manager (`manager_id` is NULL). LEFT JOIN keeps them; an inner join would drop them.

## The two rules for joins

1. **The join type decides whether unmatched rows survive.** JOIN drops them; LEFT JOIN keeps the left ones, with NULLs on the right.
2. **The uniqueness of the right-hand column decides whether matched rows multiply.** Unique → the row count stays the same. Not unique → rows get copied, and any SUM over the left table is **inflated**.

They're independent: rule 1 is your choice; rule 2 is a property of the data you **check before running**.

> **Check:** one practice query in my notes reads `JOIN departments d as e.department_id = d.department_id` and has no comma between `d.revenue` and `e.salary`. It should be:
>
> `SELECT d.revenue, e.salary FROM employees e JOIN departments d ON e.department_id = d.department_id;`

## Quick reference

| Term | In one line |
|---|---|
| `JOIN … ON` | stitch rows where the key matches |
| `LEFT JOIN` | keep all left rows; missing right = NULL |
| `… WHERE right.key IS NULL` | find rows with no match |
| Self join | same table twice, two aliases |
| Row explosion | a non-unique join key copies rows and inflates sums |
