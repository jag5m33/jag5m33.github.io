# Aliases & calculated columns

An alias is a nickname: for a column (to make the output readable) or for a table (to make the query shorter).

**On this page**

- Column aliases
- Calculated columns
- Aliases with spaces
- Table aliases
- Where you can and can't use an alias

## Column aliases

> **Definition:** `AS` renames a column in the output.

```sql
SELECT column_name AS alias_name
FROM table_name;
```

## Calculated columns

Create new columns from maths on existing ones:

```sql
SELECT
    product_name,
    unit_price * (1 - discount)  AS final_price,
    units_in_stock * unit_price  AS inventory_value
FROM products;
```

> **Check:** in my first copy of this query there's a comma after `inventory_value`, just before `FROM`. That trailing comma is a syntax error.

## Aliases with spaces

If the alias has spaces or special characters, put it in quotes. `||` glues text together (use it instead of CONCAT):

```sql
SELECT first_name || ' ' || last_name AS "Full Name",
       email AS "Contact Email"
FROM employees;
```

> **Check:** my notes use single quotes (`AS 'Full Name'`). SQLite and MySQL accept that, but standard SQL uses **double** quotes for names and single quotes for text values. Double quotes work everywhere.

## Table aliases

Simplify table names, especially in joins:

```sql
SELECT u.user_id, u.username, u.email FROM Users AS u;

SELECT c.customer_name, o.order_date, p.product_name   -- pick the columns you want from the matched rows
FROM customers AS c                                    -- start with customers (nickname c)
JOIN orders AS o                                       -- bring in orders (nickname o)
  ON c.customer_id = o.customer_id                     -- match rows where the customer_id is the same
JOIN products AS p
  ON o.product_id = p.product_id;
```

> **Check:** in my notes the comment on the `ON` line repeats "bring in the orders table". The `ON` line is the **matching rule**: pair each customer with the orders that have the same customer_id. Also, `p.product_name` needs the products table joined in (added above).

## Where you can and can't use an alias

Aliases are born in SELECT (step 5 of the [run order](#/t/sqlbasics)), so WHERE (step 2) can't see them:

```sql
-- this will NOT work
SELECT unit_price * quantity AS total_cost
FROM order_details
WHERE total_cost > 1000;

-- this WILL work: repeat the calculation
SELECT unit_price * quantity AS total_cost
FROM order_details
WHERE unit_price * quantity > 1000;
```

ORDER BY runs after SELECT, so `ORDER BY total_cost` **is** fine.

## Quick reference

| Term | In one line |
|---|---|
| `col AS name` | rename a column in the output |
| `a * b AS c` | calculated column |
| `"Full Name"` | alias with spaces: double quotes |
| `FROM table AS t` | table nickname, then `t.column` |
| Alias in WHERE | not allowed: repeat the expression |
