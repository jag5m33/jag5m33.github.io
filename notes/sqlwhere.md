# Filtering with WHERE

`WHERE` filters the results of a query: it sets a condition rows must satisfy to be included.

**On this page**

- WHERE and comparison operators
- LIKE: pattern matching
- BETWEEN and IN
- NULL: IS NULL / IS NOT NULL
- AND, OR, NOT and the order they run
- Subqueries inside WHERE

## WHERE

> **Definition:** `WHERE` keeps only the rows where the condition is **TRUE**.

```sql
SELECT column1, column2
FROM table_name
WHERE condition;

SELECT * FROM Videos WHERE views > 1000 AND user_id = 1;
SELECT * FROM Users  WHERE NOT username = 'alice';
```

## LIKE

> **Definition:** `LIKE` checks whether text matches a pattern. `%` means "any characters (or none) here".

```sql
SELECT * FROM Videos WHERE title LIKE '%Cat%' OR title LIKE '%Dog%';
```

- `'%Cat%'`: "Cat" anywhere in the title, with any words before or after.
- `'Cat%'`: starts with Cat, then anything.
- `'%tutorial'`: ends with tutorial.

> **Check:** in my notes one pattern is written `"%tutorial$"`. In SQL the wildcard is `%`, not `$`, and text goes in **single** quotes: `'%tutorial%'`.

## BETWEEN and IN

> **Definition:** `BETWEEN` tests whether a value is inside a range, **including** both ends. `IN` checks whether a value matches anything in a list (or a subquery's result).

```sql
WHERE upload_date BETWEEN '2023-02-01' AND '2023-02-10'   -- dates, numbers, even text

WHERE user_id IN (1, 2, 3)          -- instead of user_id = 1 OR user_id = 2 OR user_id = 3

-- pairs of values
WHERE (user_id, video_id) IN ((1, 100), (2, 101), (3, 102))
```

## NULL

> **Definition:** `NULL` means **unknown or missing**. You can't test it with `=`: use `IS NULL` / `IS NOT NULL`.

```sql
SELECT * FROM Interactions WHERE comment_text IS NULL;       -- no comment
SELECT * FROM Interactions WHERE comment_text IS NOT NULL;   -- has a comment
```

## AND, OR, NOT

- **AND**: combines conditions; **all** must be TRUE for the row to be returned.
- **OR**: **any** condition being TRUE is enough.
- **NOT**: reverses the condition that follows it (TRUE ↔ FALSE).

```sql
SELECT * FROM Videos
WHERE views > 1000
  AND user_id = 1
  AND upload_date >= '2023-01-01'
  AND title LIKE '%tutorial%';

SELECT * FROM Videos WHERE views > 1000 AND title NOT LIKE '%tutorial%';
```

### How AND handles NULL

| A | B | A AND B |
|---|---|---|
| TRUE | TRUE | TRUE |
| TRUE | FALSE | FALSE |
| FALSE | FALSE | FALSE |
| TRUE | NULL | **NULL** (unknown) |
| FALSE | NULL | **FALSE** (false regardless of the NULL) |
| NULL | NULL | NULL |

`WHERE x AND y` only returns the **TRUE** results: rows that come out FALSE **or NULL** are both dropped.

```sql
SELECT * FROM Users WHERE username IS NOT NULL AND join_date > '2023-01-02';
-- users with a NULL join_date are excluded, even though we don't know when they joined
```

### AND runs before OR

> **Where it goes wrong:** AND has **higher priority** than OR, so it's evaluated first.

```sql
-- what you wrote:
WHERE title LIKE '%tutorial%' OR title LIKE '%guide%' AND views > 1000
-- what SQL runs:
WHERE title LIKE '%tutorial%' OR (title LIKE '%guide%' AND views > 1000)
```

**Always use parentheses with OR.** They make the order explicit and prevent logic errors:

```sql
-- tutorials with high views OR guides with medium views
SELECT * FROM Videos
WHERE (title LIKE '%tutorial%' AND views > 1000)
   OR (title LIKE '%guide%'    AND views > 500);
```

## Subqueries inside WHERE

A query can run inside the condition, and `IN` checks against its results:

```sql
SELECT * FROM Videos
WHERE user_id IN (
    SELECT user_id
    FROM Users
    WHERE join_date >= '2023-01-05'
);
```

> **Check:** in my notes this subquery is missing its closing `)`.

Performance notes:

- IN with a very long list (hundreds or thousands of values) can be slow: use a temporary table or a JOIN instead.
- The column used with IN should be **indexed** on large tables.
- For very large data, `JOIN` or `EXISTS` can beat `IN (subquery)`. Test with `EXPLAIN`.

## Quick reference

| Term | In one line |
|---|---|
| `WHERE` | keep rows where the condition is TRUE |
| `LIKE '%x%'` | text contains x (`%` = any characters) |
| `BETWEEN a AND b` | in the range, both ends included |
| `IN (…)` | matches anything in the list / subquery |
| `IS NULL` | the only way to test for missing values |
| AND vs OR | AND runs first: use brackets |
