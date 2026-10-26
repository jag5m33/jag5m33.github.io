# jag5m33.github.io

My portfolio and study notes: a rotating sphere of dots that bursts open into a spider-web map. My notes fan out across the left half, numbered in reading order; my projects and the models inside them fan out across the right half. Dashed threads join each project and model to the notes it uses (hover to light them up). On phones the map becomes a numbered list.

Live at **https://jag5m33.github.io**

## How it's put together

| File / folder | What it is |
|---|---|
| `index.html` | The page skeleton: sphere, map, reader |
| `orb.js` | The dotted sphere on the landing screen and the small one at the centre of the map (dot count, colours and spin speed are at the top) |
| `style.css` | All colours and layout (colours are at the top) |
| `script.js` | The map, and the chapter/project pages |
| `data.json` | **The map itself**: every topic and every project |
| `notes/<id>.md` | One chapter per topic, in Markdown |
| `figures/` | Chart images used in the chapters (SVG) |
| `figures/_src/` | The Python scripts that draw those charts |
| `.nojekyll` | Tells GitHub Pages to serve the Markdown files as they are |

## Adding a topic

1. Add it to `data.json` under the right branch, e.g. `{ "id": "kmeans", "label": "K-means" }`.
   Add `"soon": true` to show it as a placeholder.
2. Write `notes/kmeans.md`.
3. Push.

The numbers on the map are worked out automatically from the order in `data.json`, so put a topic where it belongs in the learning sequence and everything after it renumbers itself. The "previous / next" links at the bottom of each chapter follow the same order.

## Writing a chapter

- `# Title`, then one sentence summary underneath.
- Maths: `$inline$` and `$$display$$` (KaTeX). A literal dollar sign is `\$`.
- Code: fenced blocks with the language, e.g. ```` ```python ````.
- Figure: `![Caption.](figures/name.svg)` on its own line.
- Callout: `> **Key idea:** ...`
- Link to another chapter: `[Bias & variance](#/t/biasvar)`. Link to a project: `#/p/acc`.

## Adding a project

Add an entry to `projects` in `data.json`:

- `repo` (owner/name) and `code` (file paths to show in the Code tab)
- `models`: the models inside the project, drawn as branches on the right half, e.g.
  `{ "id": "legal-nb", "label": "Complement Naive Bayes", "topic": "naivebayes", "status": "in progress" }`.
  `topic` draws a thread to that chapter; `status` is optional and shows the model as dashed until it's done.
- `uses`: other topic ids the project relies on (EDA, cleaning…), drawn as threads from the project
- `metrics` and `results` (image URLs) for the Results tab

To change the shape of the web, edit the layout settings at the top of `script.js` (`LEFT_SPAN`, `RING`, `LEAF_R`, `SQUASH`…).

## Redrawing a figure

```bash
pip install matplotlib numpy scipy
python3 figures/_src/<chapter>.py
```

## Previewing locally

Open the folder in VS Code and use **Live Server**. Opening `index.html` by double-clicking won't work, because browsers block the page from loading `data.json` and the notes from disk.
