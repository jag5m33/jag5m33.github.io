# Sorting with ORDER BY

`ORDER BY` organises the rows of the result in a specific sequence, sorting by one or more columns, so the results are easier to analyse.

**On this page**

- Basic syntax, ASC and DESC
- Sorting by several columns
- Sorting by position or a calculated value
- NULLs and text case

## Basic syntax

> **Definition:** `ORDER BY column ASC` sorts smallest → largest (A → Z, oldest → newest). `DESC` sorts the other way. ASC is the default.

```sql
SELECT column1, column2
FROM table_name
ORDER BY column1 [ASC|DESC], column2 [ASC|DESC];

SELECT username, join_date FROM Users ORDER BY join_date DESC;   -- newest first
SELECT title, views FROM Videos ORDER BY title ASC;              -- alphabetical
```

## Several columns

SQL applies each ORDER BY column **in the sequence you list them**: the second only breaks ties in the first.

```sql
SELECT video_id, user_id, interaction_type, timestamp
FROM Interactions
ORDER BY video_id ASC, timestamp DESC;   -- by video, then newest first within each video
```

## By position, or by a calculated value

```sql
SELECT username, email, join_date FROM Users ORDER BY 3 DESC;    -- 3 = the third column, join_date

SELECT video_id, title, views FROM Videos ORDER BY LENGTH(title) ASC;   -- by title length, not A-Z
```

ORDER BY is one of the **last** clauses to run (after FROM, WHERE, GROUP BY, HAVING and SELECT), so you **can** use aliases made in SELECT:

```sql
SELECT i.interaction_type, v.title AS video_title, i.timestamp, i.comment_text
FROM Interactions i
JOIN Videos v ON i.video_id = v.video_id
WHERE i.user_id = 3
ORDER BY i.timestamp DESC;    -- most recent first
```

## NULLs and text case

```sql
-- rows WITH a comment first, then the NULLs
SELECT interaction_id, user_id, video_id, comment_text
FROM Interactions
ORDER BY comment_text IS NULL ASC,   -- FALSE (0) comes before TRUE (1)
         comment_text ASC;
```

Sorting text can be tricky because of case: in some databases "Z" comes before "a". For case-insensitive sorting:

```sql
SELECT username FROM Users ORDER BY LOWER(username) ASC;
```

> **Check:** small typos in my notes: `ORDERBY` should be `ORDER BY`, `HWERE` → `WHERE`, `interations` → `Interactions`.

## Quick reference

| Term | In one line |
|---|---|
| `ASC` / `DESC` | ascending (default) / descending |
| Several columns | sorted by the first, ties broken by the next |
| `ORDER BY 3` | sort by the 3rd selected column |
| `col IS NULL ASC` | push NULLs to the end |
| `LOWER(col)` | case-insensitive text sort |
