<div align="center">

# 📱 Google Play Store — Data Analysis & Interactive Dashboard

Data cleaning, feature engineering, VADER sentiment analysis, and a 16-chart interactive Plotly dashboard, built end-to-end from the Google Play Store apps dataset.

[![Live Demo](https://img.shields.io/badge/demo-live-brightgreen?style=for-the-badge&logo=googlechrome&logoColor=white)](https://keshav-pro1.github.io/Google_Analytics-Project/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.18-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/python/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

**[▶ Open the live dashboard](https://keshav-pro1.github.io/Google_Analytics-Project/output/dashboard.html)** · **[📂 Browse the repo](https://github.com/keshav-pro1/Google_Analytics-Project)** · **[📓 View the notebook](notebook/creation_of_googleplaystore_da.ipynb)**

</div>

<br>

![Dashboard preview](assets/dashboard-preview.png)

<br>

## Overview

This project takes the raw Google Play Store export (~10.8K app listings, ~64K user reviews) and turns it into a cleaned dataset, a sentiment-scored review set, and a dark-themed interactive dashboard — all built with pandas, Plotly, and NLTK's VADER sentiment model. It was built as a data analytics capstone project, so on top of the core dashboard it also includes six additional filtered "task" visualizations, each intentionally gated to only render during a specific hour of the day (IST).

## Key findings

| Metric | Value |
|---|---|
| Cleaned app records | **8,892** (from 10,841 raw rows) |
| Categories | **33** |
| Free apps | **93.1%** |
| Most common category | **FAMILY** |
| Category with most installs | **GAME** |
| Median app rating | **4.3 / 5** |
| Apps with matched reviews | **1,020** |
| Total review rows processed | **64,295** |

Paid apps show a tighter, higher-median rating spread than free apps — users appear to hold paid apps to a higher bar, and are less forgiving when they miss it.

## What's on the dashboard

**10 core charts** — top categories, free vs. paid split, rating distribution, review sentiment breakdown, installs by category, update frequency by year, revenue by category, top genres, rating vs. last-update-date, and paid-vs-free rating spread.

**6 filtered "task" charts** — each answers a more specific question with its own filter logic, and each is only visible on the page during a fixed IST hour window:

| # | Chart | Question it answers | Visible (IST) | Link |
|---|---|---|---|---|
| 1 | Grouped bar — avg rating & total reviews | Top categories for Jan-updated, rating ≥ 4, size ≥ 10MB apps | 3 PM – 5 PM | embedded in [dashboard](https://keshav-pro1.github.io/Google_Analytics-Project/output/dashboard.html) |
| 2 | Choropleth — installs by country | Where are the top 5 categories (excl. A/C/G/S) installed most? | 6 PM – 8 PM | [open](https://keshav-pro1.github.io/Google_Analytics-Project/output/task2_choropleth_installs.html) |
| 3 | Dual-axis bar + line | Avg installs vs. avg revenue, Free vs. Paid | 1 PM – 2 PM | [open](https://keshav-pro1.github.io/Google_Analytics-Project/output/task3_dual_axis_installs_revenue.html) |
| 4 | Shaded time series | Monthly install trend, >20% MoM growth highlighted | 6 PM – 9 PM | [open](https://keshav-pro1.github.io/Google_Analytics-Project/output/task4_timeseries_installs_growth.html) |
| 5 | Bubble chart | App size vs. rating, bubble size = installs | 5 PM – 7 PM | [open](https://keshav-pro1.github.io/Google_Analytics-Project/output/task5_bubble_size_rating.html) |
| 6 | Stacked area | Cumulative installs over time by category | 4 PM – 6 PM | [open](https://keshav-pro1.github.io/Google_Analytics-Project/output/task6_stacked_area_cumulative_installs.html) |

Outside its window, a chart's container is fully rendered but hidden by a small `display:none` toggle — that's by design, not a bug. Reviewers checking outside those hours will just see an empty slot.

<details>
<summary><strong>Direct links to all 10 core charts</strong></summary>
<br>

- [Top Categories](https://keshav-pro1.github.io/Google_Analytics-Project/output/Category%20Graph%201.html)
- [Free vs. Paid](https://keshav-pro1.github.io/Google_Analytics-Project/output/Type%20Graph%201.html)
- [Rating Distribution](https://keshav-pro1.github.io/Google_Analytics-Project/output/Rating%20Graph%203.html)
- [Sentiment Distribution](https://keshav-pro1.github.io/Google_Analytics-Project/output/Sentiment%20Graph%204.html)
- [Installs by Category](https://keshav-pro1.github.io/Google_Analytics-Project/output/Installs%20Graph%205.html)
- [Updates per Year](https://keshav-pro1.github.io/Google_Analytics-Project/output/updates_per_year.html)
- [Revenue by Category](https://keshav-pro1.github.io/Google_Analytics-Project/output/Revenue%20graph%207.html)
- [Top Genres](https://keshav-pro1.github.io/Google_Analytics-Project/output/Genres%20graph%208.html)
- [Rating vs. Last Update](https://keshav-pro1.github.io/Google_Analytics-Project/output/Update%20Graph%209.html)
- [Paid vs. Free Rating Spread](https://keshav-pro1.github.io/Google_Analytics-Project/output/Paid%20Free%20Graph%2010.html)

</details>

## Tech stack

`Python` · `pandas` · `NumPy` · `Plotly` · `NLTK (VADER)` · `TextBlob` · `scikit-learn` · `Jupyter`

## Project structure

```
Google_Analytics-Project/
├── index.html                 # redirects Pages root → output/dashboard.html
├── assets/
│   └── dashboard-preview.png  # screenshot used above
├── data/
│   ├── Play Store Data.csv    # ~10.8K app listings, 13 columns
│   └── User Reviews.csv       # ~64K user reviews, 5 columns
├── notebook/
│   └── creation_of_googleplaystore_da.ipynb   # full analysis, executed with outputs
├── src/
│   └── generate_dashboard.py  # same pipeline as the notebook, as one script
├── output/
│   ├── dashboard.html         # main dashboard — 10 charts + Task 1
│   ├── task2_choropleth_installs.html
│   ├── task3_dual_axis_installs_revenue.html
│   ├── task4_timeseries_installs_growth.html
│   ├── task5_bubble_size_rating.html
│   ├── task6_stacked_area_cumulative_installs.html
│   └── *.html                 # each of the 10 core charts, standalone
├── requirements.txt
└── README.md
```

## Running it locally

```bash
git clone https://github.com/keshav-pro1/Google_Analytics-Project.git
cd Google_Analytics-Project
pip install -r requirements.txt
python -m nltk.downloader vader_lexicon punkt

# regenerate everything as a script
python src/generate_dashboard.py

# — or — open and re-run the notebook
jupyter notebook notebook/creation_of_googleplaystore_da.ipynb
```

`src/generate_dashboard.py` loads Plotly from a CDN rather than inlining it into every file, so `output/` stays under 2MB total. The notebook keeps Plotly fully inlined per chart (larger, but works fully offline).

## Data source

`Play Store Data.csv` and `User Reviews.csv` are the standard Kaggle "Google Play Store Apps" dataset.

## License

Released under the [MIT License](LICENSE).

---

<div align="center">

Built by [Keshav](https://github.com/keshav-pro1)

</div>
