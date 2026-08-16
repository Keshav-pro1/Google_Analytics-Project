"""
Google Play Store — Data Cleaning, EDA, Sentiment Analysis & Interactive Dashboard
-----------------------------------------------------------------------------------
This script reproduces the analysis from `creation_of_googleplaystore_da.ipynb`
so it can be run outside of Google Colab (no `files.upload()` prompts).

It:
  1. Loads and cleans the two source CSVs (Play Store Data.csv, User Reviews.csv)
  2. Engineers features (Log_Installs, Log_Reviews, Rating_Group, Revenue, Year)
  3. Runs VADER sentiment scoring on user reviews
  4. Builds 10 base Plotly charts + 6 extra "Task" charts (each with a
     JS time-gate that only shows the chart during a specific IST hour window,
     exactly as in the original notebook)
  5. Assembles everything into a single self-contained dashboard: output/dashboard.html

Run with:  python src/generate_dashboard.py
"""

import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Use the CDN build of plotly.js instead of inlining it in every single chart.
# The original notebook used include_plotlyjs='inline' (fine in Colab, but it
# bloats each exported file with a ~4.5MB copy of plotly.js). 'cdn' keeps the
# dashboard small and fast to host/share.
PLOTLYJS_MODE = "cdn"

nltk.download("vader_lexicon", quiet=True)

APPS_CSV = os.path.join(DATA_DIR, "Play Store Data.csv")
REVIEWS_CSV = os.path.join(DATA_DIR, "User Reviews.csv")

plot_width, plot_height = 400, 300
plot_containers = ""


def save_plot_as_html(fig, filename, insights):
    """Append the plot to the running dashboard HTML and also save it standalone."""
    global plot_containers
    filepath = os.path.join(OUTPUT_DIR, filename)
    html_content = pio.to_html(fig, full_html=False, include_plotlyjs=PLOTLYJS_MODE)
    plot_containers += f"""<div class="plot-container" id='{filename}' onclick="openPlot('{filename}')">
  <div class="plot"> {html_content}</div> <div class='insights'>{insights}</div> </div>"""
    fig.write_html(filepath, full_html=False, include_plotlyjs=PLOTLYJS_MODE)


def convert_size(size):
    size = str(size)
    if "M" in size:
        return float(size.replace("M", ""))
    elif "k" in size:
        return float(size.replace("k", "")) / 1024
    return np.nan


def rating_group(rating):
    if rating >= 4:
        return "Top rated app"
    elif rating >= 3:
        return "Above average"
    elif rating >= 2:
        return "Average"
    return "Below average"


# ---------------------------------------------------------------------------
# 1. Load + clean
# ---------------------------------------------------------------------------
print("Loading data...")
apps_df = pd.read_csv(APPS_CSV)
reviews_df = pd.read_csv(REVIEWS_CSV)

apps_df = apps_df.dropna(subset=["Rating"])
for column in apps_df.columns:
    apps_df[column] = apps_df[column].fillna(apps_df[column].mode()[0])
apps_df.drop_duplicates(inplace=True)
apps_df = apps_df[apps_df["Rating"] <= 5]
reviews_df.dropna(subset=["Translated_Review"], inplace=True)

apps_df["Installs"] = apps_df["Installs"].str.replace(",", "").str.replace("+", "").astype(int)
apps_df["Price"] = apps_df["Price"].str.replace("$", "", regex=False).astype(float)
apps_df["Size"] = apps_df["Size"].apply(convert_size)

merged_df = pd.merge(apps_df, reviews_df, on="App", how="inner")

# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------
apps_df["Log_Installs"] = np.log(apps_df["Installs"])
apps_df["Reviews"] = apps_df["Reviews"].astype(int)
apps_df["Log_Reviews"] = np.log(apps_df["Reviews"])
apps_df["Rating_Group"] = apps_df["Rating"].apply(rating_group)
apps_df["Revenue"] = apps_df["Price"] * apps_df["Installs"]
apps_df["Last Updated"] = pd.to_datetime(apps_df["Last Updated"], errors="coerce")
apps_df["Year"] = apps_df["Last Updated"].dt.year

# ---------------------------------------------------------------------------
# 3. Sentiment analysis (VADER)
# ---------------------------------------------------------------------------
print("Running sentiment analysis...")
sia = SentimentIntensityAnalyzer()
reviews_df["Sentiment_Score"] = reviews_df["Translated_Review"].apply(
    lambda x: sia.polarity_scores(str(x)).get("compound", 0.0)
)

# ---------------------------------------------------------------------------
# 4. Core dashboard charts (Fig 1-10)
# ---------------------------------------------------------------------------
print("Building charts...")

common_layout = dict(
    plot_bgcolor="black",
    paper_bgcolor="black",
    font_color="white",
    title_font={"size": 16},
    xaxis=dict(title_font={"size": 12}),
    yaxis=dict(title_font={"size": 12}),
    margin=dict(l=10, r=10, t=30, b=10),
)

# Fig1 - Top categories
category_count = apps_df["Category"].value_counts().nlargest(10)
fig1 = px.bar(
    x=category_count.index, y=category_count.values,
    labels={"x": "Category", "y": "Count"}, title="Top Categories on Play Store",
    color=category_count.index, color_discrete_sequence=px.colors.sequential.Plasma,
    width=plot_width, height=plot_height,
)
fig1.update_layout(**common_layout)
fig1.update_traces(marker=dict(line=dict(color="white", width=1)))
save_plot_as_html(fig1, "Category Graph 1.html", "The top categories on the play store are.")

# Fig2 - Free vs Paid
type_count = apps_df["Type"].value_counts()
fig2 = px.pie(
    values=type_count.values, names=type_count.index, title="Type Analysis",
    color=type_count.index, color_discrete_sequence=px.colors.sequential.Plasma,
    width=plot_width, height=plot_height,
)
fig2.update_layout(**common_layout)
save_plot_as_html(fig2, "Type Graph 1.html",
                   "Most apps on the playstore are free, indicating a strategy to attract users first and monetize through ads.")

# Fig3 - Rating distribution
fig3 = px.histogram(
    apps_df, x="Rating", nbins=20, title="Rating Distribution",
    color_discrete_sequence=[px.colors.sequential.Plasma[5]], labels={"Rating": "Rating"},
    width=plot_width, height=plot_height,
)
fig3.update_layout(**common_layout)
save_plot_as_html(fig3, "Rating Graph 3.html",
                   "Ratings are skewed towards higher values, suggesting that most apps are rated favourably by users.")

# Fig4 - Review sentiment distribution
sentiment_count_df = reviews_df["Sentiment"].value_counts()
fig4 = px.bar(
    x=sentiment_count_df.index, y=sentiment_count_df.values,
    labels={"x": "Sentiment Score", "y": "Count"}, title="Sentiment Distribution",
    color=sentiment_count_df.index, color_discrete_sequence=px.colors.sequential.RdPu,
    width=plot_width, height=plot_height,
)
fig4.update_layout(**common_layout)
save_plot_as_html(fig4, "Sentiment Graph 4.html",
                   "Sentiments in reviews show a mix of positive and negative feedback, with a slight lean towards positive sentiments.")

# Fig5 - Installs by category
installs_by_category = apps_df.groupby("Category")["Installs"].sum().nlargest(10)
fig5 = px.bar(
    x=installs_by_category.index, y=installs_by_category.values, orientation="h",
    labels={"x": "Total Installs", "y": "Category"}, title="Installs by Category",
    color=installs_by_category.index, color_discrete_sequence=px.colors.sequential.Plasma,
    width=plot_width, height=plot_height,
)
fig5.update_layout(**common_layout)
save_plot_as_html(fig5, "Installs Graph 5.html",
                   "The categories with the most installs are social and communication apps, reflecting their broad appeal.")

# Fig6 - Updates per year
updates_per_year = apps_df["Last Updated"].dt.year.value_counts().sort_index()
fig6 = px.line(
    x=updates_per_year.index, y=updates_per_year.values,
    labels={"x": "Year", "y": "Count"}, title="Updates per Year",
    color_discrete_sequence=["#AB63FA"], width=plot_width, height=plot_height,
)
fig6.update_layout(**common_layout)
save_plot_as_html(fig6, "updates_per_year.html",
                   "Updates have been increasing over the years, showing that developers are actively maintaining and improving their apps.")

# Fig7 - Revenue by category
revenue_by_category = apps_df.groupby("Category")["Revenue"].sum().nlargest(10)
fig7 = px.bar(
    x=revenue_by_category.index, y=revenue_by_category.values,
    labels={"x": "Category", "y": "Revenue"}, title="Revenue by Category",
    color=revenue_by_category.index, color_discrete_sequence=px.colors.sequential.Greens,
    width=plot_width, height=plot_height,
)
fig7.update_layout(**common_layout)
save_plot_as_html(fig7, "Revenue graph 7.html",
                   "Categories such as Business and Productivity lead in revenue generation, indicating their monetization strength.")

# Fig8 - Top genres
genre_counts = apps_df["Genres"].str.split(";", expand=True).stack().value_counts().nlargest(10)
fig8 = px.bar(
    x=genre_counts.index, y=genre_counts.values,
    labels={"x": "Genre", "y": "Count"}, title="Top genres",
    color=revenue_by_category.index, color_discrete_sequence=px.colors.sequential.Greens,
    width=plot_width, height=plot_height,
)
fig8.update_layout(**common_layout)
save_plot_as_html(fig8, "Genres graph 8.html",
                   "Action and Casual genres are the most common, reflecting user preference for action and easy-to-play games.")

# Fig9 - Update date vs rating
fig9 = px.scatter(
    apps_df, x="Last Updated", y="Rating", color="Type",
    title="Impact of Last Update on Rating", color_discrete_sequence=px.colors.qualitative.Vivid,
    width=plot_width, height=plot_height,
)
fig9.update_layout(**common_layout)
save_plot_as_html(fig9, "Update Graph 9.html",
                   "The scatter plot shows a weak correlation between the last update and ratings, suggesting that more frequent updates don't always result in better ratings.")

# Fig10 - Paid vs free ratings
fig10 = px.box(
    apps_df, x="Type", y="Rating", color="Type",
    title="Rating for Paid vs Free Apps", color_discrete_sequence=px.colors.qualitative.Pastel,
    width=plot_width, height=plot_height,
)
fig10.update_layout(**common_layout)
save_plot_as_html(fig10, "Paid Free Graph 10.html",
                   "Paid apps generally have higher ratings compared to free apps, suggesting that users expect higher quality from apps they pay for.")

# ---------------------------------------------------------------------------
# 5. Assemble base dashboard shell
# ---------------------------------------------------------------------------
dashboard_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Google Play Store Review Analytics</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            background-color: #333;
            color: #fff;
            margin: 0;
            padding: 0;
        }}
        .header {{
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
            background-color: #444;
        }}
        .header img {{
            margin: 0 10px;
            height: 50px;
        }}
        .container {{
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            padding: 20px;
        }}
        .plot-container {{
            border: 2px solid #555;
            margin: 10px;
            padding: 10px;
            width: {plot_width}px;
            height: {plot_height}px;
            overflow: hidden;
            position: relative;
            cursor: pointer;
        }}
        .insights {{
            display: none;
            position: absolute;
            right: 10px;
            top: 10px;
            background-color: rgba(0,0,0,0.7);
            padding: 5px;
            border-radius: 5px;
            color: #fff;
        }}
        .plot-container:hover .insights {{
            display: block;
        }}
    </style>
    <script>
        function openPlot(filename) {{
            window.open(filename, '_blank');
        }}
    </script>
</head>
<body>
    <div class="header">
        <img src="https://images.seeklogo.com/logo-png/62/1/google-new-logo-png_seeklogo-622426.png" alt="Google Logo">
        <h1>Google Play Store Reviews Analytics</h1>
        <img src="https://www.gstatic.com/marketing-cms/assets/images/15/b9/77649f194169be94fc4631a785bc/play-symbol.webp=n-w963-h543-fcrop64=1,380c0000c841ffff-rw" alt="Google Play Store Logo">
    </div>
    <div class="container">
        {plots}
    </div>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# 6. "Task" charts — each is time-gated to a specific IST hour window via JS,
#    exactly as specced in the original notebook. They are appended to the
#    main dashboard's plot_containers (Task 1) and/or saved standalone
#    (Tasks 2-6), matching the original notebook's behaviour.
# ---------------------------------------------------------------------------
print("Building time-gated task charts...")


def time_gate_script(element_id, start_hour, end_hour, fn_suffix):
    return f"""
<script>
  function checkVisibility_{fn_suffix}() {{
    const el = document.getElementById("{element_id}");
    if (!el) return;
    const istHour = parseInt(new Intl.DateTimeFormat('en-US', {{
        hour: 'numeric', hour12: false, timeZone: 'Asia/Kolkata'
    }}).format(new Date()));
    el.style.display = (istHour >= {start_hour} && istHour < {end_hour}) ? 'block' : 'none';
  }}
  window.addEventListener('load', checkVisibility_{fn_suffix});
  setInterval(checkVisibility_{fn_suffix}, 60000);
</script>
"""


# --- Task 1: Avg rating & total reviews for top categories (Jan updates, rating>=4, size>=10MB) ---
# Visible 3 PM - 5 PM IST. Appended into the MAIN dashboard.
filtered_apps_df = apps_df[
    (apps_df["Last Updated"].dt.month == 1)
    & (apps_df["Rating"] >= 4.0)
    & (apps_df["Size"].notna())
    & (apps_df["Size"] >= 10)
]
top_10_categories_by_installs = (
    filtered_apps_df.groupby("Category")["Installs"].sum().nlargest(10).index
)
final_plot_data = filtered_apps_df[filtered_apps_df["Category"].isin(top_10_categories_by_installs)]
category_metrics = final_plot_data.groupby("Category").agg(
    Average_Rating=("Rating", "mean"), Total_Reviews=("Reviews", "sum")
).reset_index()
melted_metrics = category_metrics.melt(id_vars=["Category"], var_name="Metric", value_name="Value")

fig_grouped_bar = px.bar(
    melted_metrics, x="Category", y="Value", color="Metric", barmode="group",
    title="Avg Rating & Total Reviews for Top Categories (Filtered)",
    labels={"Category": "App Category", "Value": "Value", "Metric": "Metric"},
    color_discrete_map={
        "Average_Rating": px.colors.sequential.Plasma[3],
        "Total_Reviews": px.colors.sequential.Plasma[6],
    },
    width=plot_width, height=plot_height,
)
fig_grouped_bar.update_layout(**common_layout)

t1_filename = "filtered_category_metrics_grouped_bar.html"
t1_insights = (
    "Comparison of average rating and total review count for top 10 app "
    "categories by installs, filtered for January updates, rating >= 4.0, "
    "and size >= 10MB. Visible only 3 PM-5 PM IST."
)
save_plot_as_html(fig_grouped_bar, t1_filename, t1_insights)
plot_containers += time_gate_script(t1_filename, 15, 17, "t1")

# Rebuild+save the main dashboard now that Task 1 has been folded in.
dashboard_full_html = dashboard_html.format(plot_width=plot_width, plot_height=plot_height, plots=plot_containers)
with open(os.path.join(OUTPUT_DIR, "dashboard.html"), "w", encoding="utf-8") as f:
    f.write(dashboard_full_html)
print("Saved output/dashboard.html (10 core charts + Task 1, time-gated 3-5 PM IST)")


def save_standalone_task(fig, filename, insights, start_hour, end_hour, plot_w=850, plot_h=500):
    """Tasks 2-6 in the notebook are saved as their own standalone HTML pages
    (not folded into the main dashboard) — reproduced faithfully here."""
    html_content = pio.to_html(fig, full_html=False, include_plotlyjs=PLOTLYJS_MODE)
    safe_id = filename
    block = f"""
<div class="plot-container" id="{safe_id}">
  <div class="plot">{html_content}</div>
  <div class="insights">{insights}</div>
</div>
{time_gate_script(safe_id, start_hour, end_hour, filename.replace('.', '_'))}
"""
    with open(os.path.join(OUTPUT_DIR, filename), "w", encoding="utf-8") as f:
        f.write(f"<html><body style='background:black;margin:0'>{block}</body></html>")
    print(f"Saved output/{filename} (time-gated {start_hour}:00-{end_hour}:00 IST)")


# --- Task 2: Choropleth of installs by country for top 5 categories (excluding A/C/G/S) ---
# Visible 6 PM - 8 PM IST.
t2_apps_df = pd.read_csv(APPS_CSV)
t2_apps_df = t2_apps_df.dropna(subset=["Rating", "Category", "Installs"])
t2_apps_df["Installs"] = t2_apps_df["Installs"].astype(str).str.replace(",", "").str.replace("+", "")
t2_apps_df = t2_apps_df[t2_apps_df["Installs"].str.isnumeric()]
t2_apps_df["Installs"] = t2_apps_df["Installs"].astype(int)
t2_apps_df = t2_apps_df[~t2_apps_df["Category"].str.upper().str.startswith(("A", "C", "G", "S"))]
t2_top5_categories = t2_apps_df.groupby("Category")["Installs"].sum().nlargest(5).index.tolist()
t2_filtered = t2_apps_df[t2_apps_df["Category"].isin(t2_top5_categories)]

np.random.seed(42)
t2_countries = ["USA", "IND", "GBR", "DEU", "FRA", "BRA", "CAN", "AUS", "JPN", "CHN",
                 "RUS", "ZAF", "MEX", "ITA", "ESP", "KOR", "IDN", "NGA", "EGY", "ARG"]
t2_rows = []
for category in t2_top5_categories:
    total_installs = t2_filtered.loc[t2_filtered["Category"] == category, "Installs"].sum()
    weights = np.random.dirichlet(np.ones(len(t2_countries)))
    for country, w in zip(t2_countries, weights):
        t2_rows.append({"Category": category, "Country": country, "Installs": int(total_installs * w)})
t2_geo_df = pd.DataFrame(t2_rows)
t2_geo_df["Highlight"] = np.where(t2_geo_df["Installs"] > 1_000_000, "Above 1M \u2b50", "Below 1M")

fig_t2 = px.choropleth(
    t2_geo_df, locations="Country", color="Installs", hover_name="Country",
    hover_data={"Category": True, "Installs": True, "Highlight": True},
    animation_frame="Category", color_continuous_scale=px.colors.sequential.Plasma,
    title="Global Installs by Category (Top 5, excluding A/C/G/S-starting categories)",
    width=800, height=500,
)
fig_t2.update_layout(paper_bgcolor="black", plot_bgcolor="black", font_color="white",
                      geo=dict(bgcolor="black", showframe=False, showcoastlines=True))
save_standalone_task(
    fig_t2, "task2_choropleth_installs.html",
    "Global install distribution for the top 5 categories (A/C/G/S-starting categories "
    "excluded). Countries with over 1M installs are flagged in hover text. "
    "Note: no geo column exists in the source data, so installs are proportionally "
    "simulated across countries for illustration.",
    18, 20,
)

# --- Task 3: Avg installs vs avg revenue, Free vs Paid, top 3 filtered categories ---
# Visible 1 PM - 2 PM IST.
import re
t3_apps_df = pd.read_csv(APPS_CSV)
t3_apps_df = t3_apps_df.dropna(subset=["Rating", "Category", "Installs", "Price",
                                        "Android Ver", "Size", "Content Rating", "App"])
t3_apps_df["Installs"] = t3_apps_df["Installs"].astype(str).str.replace(",", "").str.replace("+", "")
t3_apps_df = t3_apps_df[t3_apps_df["Installs"].str.isnumeric()]
t3_apps_df["Installs"] = t3_apps_df["Installs"].astype(int)
t3_apps_df["Price"] = t3_apps_df["Price"].astype(str).str.replace("$", "", regex=False)
t3_apps_df["Price"] = pd.to_numeric(t3_apps_df["Price"], errors="coerce")
t3_apps_df = t3_apps_df.dropna(subset=["Price"])
t3_apps_df["Revenue"] = t3_apps_df["Price"] * t3_apps_df["Installs"]
t3_apps_df["Size"] = t3_apps_df["Size"].apply(convert_size)


def t3_parse_android_ver(v):
    match = re.search(r"(\d+(\.\d+)?)", str(v))
    return float(match.group(1)) if match else np.nan


t3_apps_df["Android_Ver_Num"] = t3_apps_df["Android Ver"].apply(t3_parse_android_ver)

t3_filtered = t3_apps_df[
    (t3_apps_df["Installs"] >= 10000)
    & (t3_apps_df["Revenue"] >= 10000)
    & (t3_apps_df["Android_Ver_Num"] > 4.0)
    & (t3_apps_df["Size"] >= 15)
    & (t3_apps_df["Content Rating"] == "Everyone")
    & (t3_apps_df["App"].str.len() <= 30)
]
t3_top3_categories = t3_filtered.groupby("Category")["Installs"].sum().nlargest(3).index.tolist()
t3_final = t3_filtered[t3_filtered["Category"].isin(t3_top3_categories)]
t3_summary = t3_final.groupby(["Category", "Type"]).agg(
    Avg_Installs=("Installs", "mean"), Avg_Revenue=("Revenue", "mean")
).reset_index()

fig_t3 = make_subplots(specs=[[{"secondary_y": True}]])
for app_type, color in [("Free", "#00CC96"), ("Paid", "#EF553B")]:
    sub = t3_summary[t3_summary["Type"] == app_type]
    fig_t3.add_trace(
        go.Bar(x=sub["Category"], y=sub["Avg_Installs"], name=f"{app_type} - Avg Installs",
               marker_color=color, opacity=0.7),
        secondary_y=False,
    )
    fig_t3.add_trace(
        go.Scatter(x=sub["Category"], y=sub["Avg_Revenue"], name=f"{app_type} - Avg Revenue",
                   mode="lines+markers", line=dict(width=3)),
        secondary_y=True,
    )
fig_t3.update_layout(
    title="Avg Installs vs Avg Revenue \u2014 Free vs Paid (Top 3 Categories, Filtered)",
    barmode="group", plot_bgcolor="black", paper_bgcolor="black", font_color="white",
    width=800, height=450,
)
fig_t3.update_yaxes(title_text="Avg Installs", secondary_y=False)
fig_t3.update_yaxes(title_text="Avg Revenue ($)", secondary_y=True)
save_standalone_task(
    fig_t3, "task3_dual_axis_installs_revenue.html",
    "Avg installs (bars, left axis) vs avg revenue (lines, right axis) for Free vs Paid "
    "apps across the top 3 filtered categories.",
    13, 14,
)

# --- Task 4: Monthly install trend by category (E/C/B), >20% MoM growth shaded ---
# Visible 6 PM - 9 PM IST.
t4_apps_df = pd.read_csv(APPS_CSV)
t4_apps_df = t4_apps_df.dropna(subset=["Rating", "Category", "Installs", "Reviews", "Last Updated", "App"])
t4_apps_df["Installs"] = t4_apps_df["Installs"].astype(str).str.replace(",", "").str.replace("+", "")
t4_apps_df = t4_apps_df[t4_apps_df["Installs"].str.isnumeric()]
t4_apps_df["Installs"] = t4_apps_df["Installs"].astype(int)
t4_apps_df["Reviews"] = pd.to_numeric(t4_apps_df["Reviews"], errors="coerce")
t4_apps_df["Last Updated"] = pd.to_datetime(t4_apps_df["Last Updated"], errors="coerce")
t4_apps_df = t4_apps_df.dropna(subset=["Last Updated", "Reviews"])

t4_apps_df = t4_apps_df[~t4_apps_df["App"].str.lower().str.startswith(("x", "y", "z"))]
t4_apps_df = t4_apps_df[t4_apps_df["Category"].str.upper().str.startswith(("E", "C", "B"))]
t4_apps_df = t4_apps_df[t4_apps_df["Reviews"] > 500]
t4_apps_df = t4_apps_df[~t4_apps_df["App"].str.upper().str.contains("S")]

t4_apps_df["Month"] = t4_apps_df["Last Updated"].dt.to_period("M").dt.to_timestamp()
t4_monthly = t4_apps_df.groupby(["Category", "Month"])["Installs"].sum().reset_index()
t4_monthly = t4_monthly.sort_values(["Category", "Month"])
t4_monthly["Pct_Growth"] = t4_monthly.groupby("Category")["Installs"].pct_change() * 100
t4_translate = {"Beauty": "\u0938\u0941\u0902\u0926\u0930\u0924\u093e (Beauty)",
                 "Business": "\u0bb5\u0ba3\u0bbf\u0b95\u0bae\u0bcd (Business)",
                 "Dating": "Dating (Partnersuche)"}
t4_monthly["Category_Display"] = t4_monthly["Category"].replace(t4_translate)

fig_t4 = go.Figure()
t4_color_cycle = px.colors.qualitative.Set2
for i, category in enumerate(t4_monthly["Category"].unique()):
    sub = t4_monthly[t4_monthly["Category"] == category].reset_index(drop=True)
    display_name = sub["Category_Display"].iloc[0]
    color = t4_color_cycle[i % len(t4_color_cycle)]
    fig_t4.add_trace(go.Scatter(x=sub["Month"], y=sub["Installs"], mode="lines+markers",
                                 name=display_name, line=dict(color=color, width=2)))
    for j in range(1, len(sub)):
        if sub.loc[j, "Pct_Growth"] > 20:
            fig_t4.add_trace(go.Scatter(
                x=[sub.loc[j - 1, "Month"], sub.loc[j, "Month"]],
                y=[sub.loc[j - 1, "Installs"], sub.loc[j, "Installs"]],
                fill="tozeroy", mode="none", fillcolor=color, opacity=0.25,
                showlegend=False, hoverinfo="skip",
            ))
fig_t4.update_layout(
    title="Installs Trend by Category (Shaded = >20% MoM Growth)",
    plot_bgcolor="black", paper_bgcolor="black", font_color="white",
    xaxis_title="Month", yaxis_title="Total Installs", width=850, height=450,
)
save_standalone_task(
    fig_t4, "task4_timeseries_installs_growth.html",
    "Monthly install trends for Beauty, Business, Dating, and other E/C/B categories. "
    "Shaded regions mark months where installs grew >20% vs the previous month.",
    18, 21,
)

# --- Task 5: App size vs rating bubble chart (Game highlighted pink) ---
# Visible 5 PM - 7 PM IST.
from textblob import TextBlob
t5_apps_df = pd.read_csv(APPS_CSV)
t5_reviews_df = pd.read_csv(REVIEWS_CSV)
t5_apps_df = t5_apps_df.dropna(subset=["Rating", "Category", "Installs", "Size", "Reviews", "App"])
t5_reviews_df = t5_reviews_df.dropna(subset=["Translated_Review", "App"])
t5_apps_df["Installs"] = t5_apps_df["Installs"].astype(str).str.replace(",", "").str.replace("+", "")
t5_apps_df = t5_apps_df[t5_apps_df["Installs"].str.isnumeric()]
t5_apps_df["Installs"] = t5_apps_df["Installs"].astype(int)
t5_apps_df["Reviews"] = pd.to_numeric(t5_apps_df["Reviews"], errors="coerce")
t5_apps_df["Size"] = t5_apps_df["Size"].apply(convert_size)
t5_apps_df = t5_apps_df.dropna(subset=["Size", "Reviews"])

t5_reviews_df["Subjectivity"] = t5_reviews_df["Translated_Review"].apply(
    lambda x: TextBlob(str(x)).sentiment.subjectivity
)
t5_app_subjectivity = t5_reviews_df.groupby("App")["Subjectivity"].mean().reset_index()
t5_merged = pd.merge(t5_apps_df, t5_app_subjectivity, on="App", how="inner")

t5_target_categories = ["GAME", "BEAUTY", "BUSINESS", "COMICS", "COMMUNICATION",
                         "DATING", "ENTERTAINMENT", "SOCIAL", "EVENTS"]
t5_final = t5_merged[
    (t5_merged["Rating"] > 3.5)
    & (t5_merged["Category"].isin(t5_target_categories))
    & (t5_merged["Reviews"] > 500)
    & (~t5_merged["App"].str.upper().str.contains("S"))
    & (t5_merged["Subjectivity"] > 0.5)
    & (t5_merged["Installs"] > 50000)
]
t5_translate = {"BEAUTY": "\u0938\u0941\u0902\u0926\u0930\u094d\u092f (Beauty)",
                 "BUSINESS": "\u0bb5\u0ba3\u0bbf\u0b95\u0bae\u0bcd (Business)",
                 "DATING": "Dating (Partnersuche)"}
t5_final = t5_final.copy()
t5_final["Category_Display"] = t5_final["Category"].replace(t5_translate)

t5_categories_display = t5_final["Category_Display"].unique().tolist()
t5_palette = px.colors.qualitative.Set2
t5_color_map = {}
palette_i = 0
for cat in t5_categories_display:
    if cat == "GAME":
        t5_color_map[cat] = "hotpink"
    else:
        t5_color_map[cat] = t5_palette[palette_i % len(t5_palette)]
        palette_i += 1

fig_t5 = px.scatter(
    t5_final, x="Size", y="Rating", size="Installs", color="Category_Display",
    color_discrete_map=t5_color_map, hover_name="App",
    hover_data=["Installs", "Reviews", "Subjectivity"],
    title="App Size vs Rating (Bubble = Installs) \u2014 Game highlighted in Pink",
    size_max=45, width=850, height=500,
)
fig_t5.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white")
save_standalone_task(
    fig_t5, "task5_bubble_size_rating.html",
    "Bubble size = installs. Rating > 3.5, reviews > 500, install count > 50k, review "
    "subjectivity > 0.5. Game category highlighted pink.",
    17, 19,
)

# --- Task 6: Cumulative installs over time by category (stacked area) ---
# Visible 4 PM - 6 PM IST.
t6_apps_df = pd.read_csv(APPS_CSV)
t6_apps_df = t6_apps_df.dropna(subset=["Rating", "Category", "Installs", "App", "Reviews", "Size", "Last Updated"])
t6_apps_df["Installs"] = t6_apps_df["Installs"].astype(str).str.replace(",", "").str.replace("+", "")
t6_apps_df = t6_apps_df[t6_apps_df["Installs"].str.isnumeric()]
t6_apps_df["Installs"] = t6_apps_df["Installs"].astype(int)
t6_apps_df["Reviews"] = pd.to_numeric(t6_apps_df["Reviews"], errors="coerce")
t6_apps_df["Last Updated"] = pd.to_datetime(t6_apps_df["Last Updated"], errors="coerce")
t6_apps_df["Size"] = t6_apps_df["Size"].apply(convert_size)
t6_apps_df = t6_apps_df.dropna(subset=["Size", "Reviews", "Last Updated"])

t6_apps_df = t6_apps_df[t6_apps_df["Rating"] >= 4.2]
t6_apps_df = t6_apps_df[~t6_apps_df["App"].str.contains(r"\d", regex=True)]
t6_apps_df = t6_apps_df[t6_apps_df["Category"].str.upper().str.startswith(("T", "P"))]
t6_apps_df = t6_apps_df[t6_apps_df["Reviews"] > 1000]
t6_apps_df = t6_apps_df[(t6_apps_df["Size"] >= 20) & (t6_apps_df["Size"] <= 80)]

t6_apps_df["Month"] = t6_apps_df["Last Updated"].dt.to_period("M").dt.to_timestamp()
t6_monthly = t6_apps_df.groupby(["Category", "Month"])["Installs"].sum().reset_index()
t6_monthly = t6_monthly.sort_values(["Category", "Month"])
t6_monthly["Cumulative_Installs"] = t6_monthly.groupby("Category")["Installs"].cumsum()
t6_monthly["Pct_Growth"] = t6_monthly.groupby("Category")["Installs"].pct_change() * 100
t6_monthly["High_Growth"] = t6_monthly["Pct_Growth"] > 25

t6_translate = {"Travel & Local": "Voyages et Local (FR)", "Productivity": "Productividad (ES)",
                 "Photography": "\u5199\u771f (JA)"}
t6_monthly["Category_Display"] = t6_monthly["Category"].replace(t6_translate)

fig_t6 = px.area(
    t6_monthly, x="Month", y="Cumulative_Installs", color="Category_Display",
    title="Cumulative Installs Over Time by Category (Stacked)",
    color_discrete_sequence=px.colors.sequential.Plasma, width=850, height=480,
)
t6_highlight = t6_monthly[t6_monthly["High_Growth"]]
fig_t6.add_trace(go.Scatter(
    x=t6_highlight["Month"], y=t6_highlight["Cumulative_Installs"], mode="markers",
    marker=dict(size=12, color="yellow", symbol="star", line=dict(color="black", width=1)),
    name=">25% MoM growth", hoverinfo="text",
    text=t6_highlight["Category_Display"] + " \u2014 " + t6_highlight["Pct_Growth"].round(1).astype(str) + "%",
))
fig_t6.update_layout(plot_bgcolor="black", paper_bgcolor="black", font_color="white",
                      xaxis_title="Month", yaxis_title="Cumulative Installs")
save_standalone_task(
    fig_t6, "task6_stacked_area_cumulative_installs.html",
    "Cumulative installs by category (T/P-starting only). Gold stars mark months where "
    "a category's installs grew >25% vs the previous month. Legend shows translated "
    "names for Travel & Local, Productivity, Photography.",
    16, 18,
)

print("\nAll done. Files written to:", OUTPUT_DIR)
for fname in sorted(os.listdir(OUTPUT_DIR)):
    fpath = os.path.join(OUTPUT_DIR, fname)
    print(f"  - {fname}  ({os.path.getsize(fpath)/1024:.0f} KB)")
