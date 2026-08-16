# 📱 Google Play Store — Data Analysis & Interactive Dashboard

Exploratory data analysis, sentiment analysis, and an interactive Plotly dashboard built from the Google Play Store apps dataset (~8.9K cleaned apps, ~64K user reviews).

## 🔗 Live dashboard

[Open the interactive dashboard](output/dashboard.html) *(replace with your GitHub Pages / hosted URL once published — see [Hosting](#-hosting-it-live) below)*

## 📋 What's in this project

- **Data cleaning**: nulls, duplicates, malformed `Installs`/`Price`/`Size` columns
- **Feature engineering**: `Log_Installs`, `Log_Reviews`, `Rating_Group`, `Revenue`, `Year`
- **Sentiment analysis**: VADER polarity scoring on ~64K translated user reviews
- **10 core charts**: category breakdown, free vs. paid split, rating distribution, review sentiment, installs by category, update frequency, revenue by category, top genres, rating vs. last-update, paid vs. free rating spread
- **6 additional "task" charts**, each filtered/aggregated differently and — as a deliberate design constraint of the assignment — only visible in the browser during a specific IST (India Standard Time) hour window via a small JS time-gate:

  | Task | Chart | Visible (IST) |
  |---|---|---|
  | 1 | Avg rating & total reviews, top categories (Jan updates, rating ≥ 4, size ≥ 10MB) | 3 PM – 5 PM |
  | 2 | Simulated choropleth of installs by country, top 5 categories | 6 PM – 8 PM |
  | 3 | Dual-axis avg installs vs. avg revenue, Free vs. Paid | 1 PM – 2 PM |
  | 4 | Monthly install trend, >20% MoM growth shaded | 6 PM – 9 PM |
  | 5 | App size vs. rating bubble chart (installs = bubble size) | 5 PM – 7 PM |
  | 6 | Cumulative installs over time, stacked area | 4 PM – 6 PM |

  Task 1 is folded into the main dashboard (`output/dashboard.html`); Tasks 2–6 are standalone pages in `output/`. Open them outside their visible window and the JS just leaves the chart hidden — that's expected, not a bug.

## 📊 A few things the data shows

- **8,892** cleaned app records, **33** categories, **64,295** raw review rows (1,020 unique apps have both listing + review data)
- **93.1%** of apps are free
- **FAMILY** is the most common category by app count; **GAME** leads in total installs
- Median app rating is **4.3** — ratings are strongly skewed toward the high end
- Paid apps tend to have a tighter, higher-median rating spread than free apps

## 🗂 Project structure

```
.
├── data/
│   ├── Play Store Data.csv       # ~10.8K app listings, 13 columns
│   └── User Reviews.csv          # ~64K user reviews, 5 columns
├── notebook/
│   └── creation_of_googleplaystore_da.ipynb   # full analysis, cell-by-cell, with outputs
├── src/
│   └── generate_dashboard.py     # same pipeline as the notebook, as one runnable script
├── output/
│   ├── dashboard.html            # main dashboard (10 charts + Task 1)
│   ├── task2_choropleth_installs.html
│   ├── task3_dual_axis_installs_revenue.html
│   ├── task4_timeseries_installs_growth.html
│   ├── task5_bubble_size_rating.html
│   ├── task6_stacked_area_cumulative_installs.html
│   └── *.html                    # each of the 10 core charts, standalone
├── requirements.txt
└── README.md
```

## ▶️ Running it yourself

```bash
git clone <your-repo-url>
cd <repo-folder>
pip install -r requirements.txt
python -m nltk.downloader vader_lexicon punkt

# Option A — regenerate everything as a script
python src/generate_dashboard.py

# Option B — open and re-run the notebook
jupyter notebook notebook/creation_of_googleplaystore_da.ipynb
```

Both produce the same charts. `src/generate_dashboard.py` is the leaner path — it loads plotly.js from a CDN rather than inlining a ~4.5MB copy into every chart, so `output/` stays under 2MB total instead of 100+MB. The notebook keeps the original `include_plotlyjs='inline'` behavior (self-contained files, larger).

## 🌐 Hosting it live

You have two easy options once this is pushed to GitHub:

**GitHub Pages** (a public URL for the dashboard):
1. Push this repo to GitHub.
2. Repo **Settings → Pages → Source**, pick the `main` branch and `/output` (or `/`) folder → **Save**.
3. GitHub gives you a URL like `https://<your-username>.github.io/<repo-name>/dashboard.html` — that's your shareable live link. It can take a minute or two to go live after the first push.

**Claude Artifact link**: if you're viewing `output/dashboard.html` as a Claude artifact in this chat, use the **Publish** button in the artifact panel to get an instant shareable public link — no GitHub required.

Remember the Task 2–6 charts are time-gated to specific IST hours (see table above), so reviewers may need to check back at the right time, or you can temporarily widen the hour ranges in `src/generate_dashboard.py` / the notebook for a demo.

## 🧰 Tech stack

Python · pandas · NumPy · Plotly · NLTK (VADER) · TextBlob · scikit-learn (imported for potential modeling, not currently used in a model) · Jupyter

## 📁 Data source

`Play Store Data.csv` and `User Reviews.csv` are the standard Kaggle "Google Play Store Apps" dataset (~10.8K app listings / ~64K reviews).

## 📝 Notes on this repo vs. the original notebook

- Colab-only cells (`google.colab.files.upload()`, inline `/content/...` HTML previews) were removed/adapted so the notebook runs anywhere, not just Colab.
- A stray `_#Fig1` line (leftover from an accidental keystroke, harmless in Colab's IPython shell but a `NameError` in a plain script) was fixed to a normal `#Fig1` comment.
- CSV paths point at `../data/...` instead of the working directory, so the notebook and script can both find the data from their own folder.
