# =====================================
# generate_dashboard.py — Version 5.1
# =====================================

import os
import json
from LDA_engine_with_BERTopic_v05 import (
    fetch_articles,
    run_topic_model,
    summarize_topic_gpt,
)

import plotly.express as px
import plotly.graph_objects as go

# -------------------------------
# STEP 1: Fetch articles
# -------------------------------
RSS_FEEDS = [
    "http://feeds.bbci.co.uk/news/rss.xml",
    "https://rss.cnn.com/rss/edition.rss",
    "https://feeds.a.dj.com/rss/RSSWorldNews.xml",
    "https://www.reutersagency.com/feed/?best-sectors=world&post_type=best",
]

print("Fetching articles...")
texts = fetch_articles(RSS_FEEDS)
texts = texts[:50]  # cap to prevent instability

if len(texts) < 10:
    texts += [" filler content "] * (10 - len(texts))
    print("Low content – padded with placeholders.")


# -------------------------------
# STEP 2: Run BERTopic
# -------------------------------
print("Running BERTopic...")
topic_model, topics = run_topic_model(texts)
topic_info = topic_model.get_topic_info()

# Filter valid topics
valid_topics = [t for t in topic_info["Topic"] if t != -1]
print(f"Detected {len(valid_topics)} valid topics.")


# -------------------------------
# STEP 3: GPT summaries
# -------------------------------
print("Generating GPT summaries...")
topic_summaries = []

for topic_id in valid_topics[:5]:  # limit to first 5 topics
    words = [w for w, _ in topic_model.get_topic(topic_id)]
    docs = [texts[i] for i, t in enumerate(topics) if t == topic_id][:3]
    summary_json = summarize_topic_gpt(topic_id, words, docs)

    topic_summaries.append({
        "topic": topic_id,
        "summary": summary_json
    })


# -------------------------------
# STEP 4: Safe visualizations
# -------------------------------
print("Building visualizations...")

if len(valid_topics) < 2:
    print("WARNING: Not enough distinct topics for visualization.")
    fig_topics = "<p><b>No topic map available (only one topic).</b></p>"
    fig_bars = "<p><b>No barchart available.</b></p>"
else:
    fig_topics = topic_model.visualize_topics(width=600, height=700)
    fig_bars = topic_model.visualize_barchart(width=600, height=700)


# -------------------------------
# STEP 5: Build HTML
# -------------------------------
print("Building HTML...")

html_output = f"""
<html>
<head>
<title>News Topic Dashboard</title>
</head>
<body>

<h1>Daily News Topics</h1>

<h2>Topic Map</h2>
{fig_topics if isinstance(fig_topics, str) else fig_topics.to_html(include_plotlyjs='cdn')}

<h2>Topic Summaries</h2>
<pre>{json.dumps(topic_summaries, indent=2)}</pre>

<h2>Topic Barchart</h2>
{fig_bars if isinstance(fig_bars, str) else fig_bars.to_html(include_plotlyjs='cdn')}

</body>
</html>
"""

os.makedirs("dashboard", exist_ok=True)

with open("dashboard/index.html", "w", encoding="utf-8") as f:
    f.write(html_output)

print("Dashboard generated successfully → dashboard/index.html")
