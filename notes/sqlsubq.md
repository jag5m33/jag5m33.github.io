# Subqueries & CTEs

Break a hard question into steps: answer a smaller question first, then query its result.

**On this page**

- Subqueries in FROM
- CTEs (WITH)
- Which to use

## Subqueries in FROM

> **Definition:** a **subquery** is a query inside another query. In FROM, its result acts like a temporary table, which needs an alias.

```sql
SELECT
    dept_summary.department_name,
    dept_summary.avg_salary
FROM (
    SELECT d.department_name, AVG(e.salary) AS avg_salary
    FROM departments d
    JOIN employees e ON d.department_id = e.department_id
    GROUP BY d.department_name
) AS dept_summary
WHERE dept_summary.avg_salary > 50000;
```

- **Step 1 (inner query):** "Calculate the average salary for each department and write the results on a temporary sticky note."
- **Step 2 (outer query):** "Now look at that sticky note and cross off any department where the average is $50,000 or lower."

Subqueries can also sit inside WHERE: see [Filtering with WHERE](#/t/sqlwhere).

## CTEs (WITH)

> **Definition:** a **CTE** (common table expression) is a named, temporary result set made with `WITH`. It exists only while that one query runs and can be referenced several times inside it.

- Like a temporary view: **not** stored in the database.
- Breaks complex queries into smaller, readable parts.
- Works with SELECT, INSERT, UPDATE, DELETE and MERGE.

```sql
WITH high_value_orders AS (
    SELECT
        customer_id,
        COUNT(*)          AS order_count,   -- count all orders for each customer_id
        SUM(total_amount) AS total_spent    -- total amount spent by each customer
    FROM orders
    GROUP BY customer_id                    -- one row per customer
    HAVING SUM(total_amount) > 10000        -- only those who spent over 10,000
)
SELECT c.customer_name, hvo.order_count, hvo.total_spent
FROM customers c
JOIN high_value_orders hvo ON c.customer_id = hvo.customer_id;
```

> **Check:** three fixes to my notes' version: (1) it sums `total_count` but filters on `total_amount`; it should be the same column. (2) There's a comma after `GROUP BY customer_id`, which breaks the query. (3) SQL comments start with `--`, not `#`.

Another one: companies with duplicate job listings.

```sql
WITH job_count AS (
    SELECT company_id, title, description, COUNT(job_id) AS job_count
    FROM job_listings
    GROUP BY company_id, title, description
)
SELECT COUNT(DISTINCT company_id) AS duplicate_companies
FROM job_count
WHERE job_count >= 2;
```

## Which to use

| | Subquery | CTE |
|---|---|---|
| Reads | inside out | top to bottom |
| Reuse | written once per use | named once, used many times |
| Best for | a short one-off step | anything with 2+ steps |

## Quick reference

| Term | In one line |
|---|---|
| Subquery in FROM | a query used as a temporary table: needs an alias |
| Subquery in WHERE | a list to check against with `IN` |
| CTE | `WITH name AS (…)`: a named step, read top to bottom |
