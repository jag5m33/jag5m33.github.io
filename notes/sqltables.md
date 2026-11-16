# Creating & changing tables

Commands that change the database itself, not just read from it.

**On this page**

- CREATE and DROP
- ALTER TABLE
- INSERT, UPDATE, DELETE

## CREATE and DROP

> **Definition:** `CREATE TABLE` makes a new table: each column gets a name, a data type and optional rules (NOT NULL, decimal places etc.). `DROP TABLE` deletes the table **and all its data**.

```sql
CREATE TABLE products (
    product_id   INTEGER PRIMARY KEY,
    name         TEXT NOT NULL,
    unit_price   DECIMAL(10, 2)
);

DROP TABLE products;
```

## ALTER TABLE

```sql
ALTER TABLE products ADD COLUMN serial_number INT;        -- add a serial number column
ALTER TABLE products ALTER COLUMN serial_number TYPE TEXT; -- change its data type to text (PostgreSQL)
```

> **Check:** in my notes it's `add serial number INT`: column names can't have spaces, so `serial_number`, and the full form is `ADD COLUMN`. Changing a type is written differently per database (`ALTER COLUMN … TYPE` in PostgreSQL, `MODIFY` in MySQL), and SQLite can't change a column's type at all.

## INSERT, UPDATE, DELETE

```sql
INSERT INTO products (product_id, name, unit_price)
VALUES (1, 'Kettle', 24.99),
       (2, 'Toaster', 19.50);      -- values must follow the column order in the first line

UPDATE products
SET unit_price = 21.00,
    name = 'Toaster XL'
WHERE product_id = 2;              -- set these columns, only on the matching rows

DELETE FROM products WHERE product_id = 2;
```

> **Where it goes wrong:** **always** specify a filtering condition on UPDATE and DELETE. Without `WHERE`, every row in the table is changed or deleted.

> **Check:** in my notes there's a colon after `INSERT INTO table_name(...)` and no comma between the value groups. It's `VALUES (…), (…)` with a comma and no colon.

## Quick reference

| Command | Does |
|---|---|
| `CREATE TABLE` | new table: names, types, rules |
| `ALTER TABLE … ADD COLUMN` | add a column |
| `DROP TABLE` | delete the table and all data |
| `INSERT INTO … VALUES` | add rows |
| `UPDATE … SET … WHERE` | change rows |
| `DELETE FROM … WHERE` | remove rows |
