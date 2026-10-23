# jag5m33.github.io

My portfolio and study notes: a rotating sphere of dots that bursts open into a mind map. Topics on the left, my GitHub projects on the right, dashed lines showing which methods each project uses.

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

## Writing a chapter

- `# Title`, then one sentence summary underneath.
- Maths: `$inline$` and `$$display$$` (KaTeX). A literal dollar sign is `\$`.
- Code: fenced blocks with the language, e.g. ```` ```python ````.
- Figure: `![Caption.](figures/name.svg)` on its own line.
- Callout: `> **Key idea:** ...`
- Link to another chapter: `[Bias & variance](#/t/biasvar)`. Link to a project: `#/p/acc`.

## Adding a project

Add an entry to `projects` in `data.json`: `repo` (owner/name), `code` (file paths to show), `uses` (topic ids, which draws the dashed links), `metrics` and `results` (image URLs).

## Redrawing a figure

```bash
pip install matplotlib numpy scipy
python3 figures/_src/<chapter>.py
```

## Previewing locally

Open the folder in VS Code and use **Live Server**. Opening `index.html` by double-clicking won't work, because browsers block the page from loading `data.json` and the notes from disk.
