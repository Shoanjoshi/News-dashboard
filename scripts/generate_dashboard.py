import os
import json
import datetime

from LDA_engine_with_BERTopic_v04 import (
    fetch_articles,
    build_bertopic_model,
    interpret_topics,
    MAX_ARTICLES_TO_FETCH,
    NUM_TOPICS_FOR_LDA,
)

OUTPUT_DIR = "dashboard"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_dashboard():
    # ------------------------------------------------------------
    # 1. Full pipeline (same as v04)
    # ------------------------------------------------------------
    print("Fetching articles...")
    articles = fetch_articles(MAX_ARTICLES_TO_FETCH)
    texts = [a["text"] for a in articles]

    print("Running BERTopic...")
    topic_model, topics, probs = build_bertopic_model(
        texts, num_topics=NUM_TOPICS_FOR_LDA
    )

    print("Generating LLM topic summaries...")
    topic_summaries_raw = interpret_topics(topic_model, texts)

    # Parse JSON outputs into Python dicts
    topic_summaries = {}
    for tid, summary_json in topic_summaries_raw.items():
        try:
            topic_summaries[tid] = json.loads(summary_json)
        except:
            topic_summaries[tid] = {
                "topic_id": tid,
                "label": f"Topic {tid}",
                "description": summary_json,
                "subthemes": []
            }

    # ------------------------------------------------------------
    # 2. Build better visualization metadata
    # ------------------------------------------------------------
    doc_info = topic_model.get_document_info(texts)
    topic_info = topic_model.get_topic_info()

    # Only real topics (no -1)
    valid_topics = topic_info[topic_info["Topic"] != -1]["Topic"].tolist()

    hover_texts = []
    custom_labels = []

    for tid in valid_topics:
        # GPT label
        label = topic_summaries[tid]["label"]

        # Top words
        top_words = topic_model.get_topic(tid)
        top_words_text = ", ".join([w[0] for w in top_words[:5]]) if top_words else ""

        # Top 3 headlines
        df_t = (
            doc_info[doc_info["Topic"] == tid]
            .sort_values("Probability", ascending=False)
            .head(3)
        )
        headlines = []
        for idx in df_t.index:
            title = articles[idx]["title"]
            if title:
                headlines.append(title)

        headlines_text = "; ".join(headlines) if headlines else "No headline"

        # Combined hover
        hover_texts.append(
            f"<b>{label}</b><br>"
            f"<b>Top words:</b> {top_words_text}<br>"
            f"<b>Headlines:</b> {headlines_text}"
        )

        custom_labels.append(label)

    # ------------------------------------------------------------
    # 3. Nicer BERTopic map (same position / layout)
    # ------------------------------------------------------------
    fig = topic_model.visualize_topics(
        custom_labels=True,
        width=900,
        height=800
    )

    # Inject hover text
    fig.data[0].text = hover_texts
    fig.data[0].hovertemplate = "%{text}<extra></extra>"

    # Convert to embeddable HTML snippet
    topic_map_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

    # ------------------------------------------------------------
    # OPTIONAL: Add barchart of top words
    # ------------------------------------------------------------
    barchart_fig = topic_model.visualize_barchart(
        top_n_topics=NUM_TOPICS_FOR_LDA,
        width=900,
        height=600
    )
    barchart_html = barchart_fig.to_html(full_html=False, include_plotlyjs=False)

    # ------------------------------------------------------------
    # 4. Build the same HTML dashboard as before
    # ------------------------------------------------------------
    timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    html = f"""
<html>
<head>
    <title>Daily News Topic Dashboard (BERTopic + GPT)</title>

    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
        }}

        .header {{
            background: #f2f2f2;
            padding: 12px;
            font-size: 15px;
            font-weight: bold;
            border-bottom: 2px solid #ccc;
        }}

        .main-container {{
            display: flex;
            height: calc(100vh - 48px);
        }}

        .left-panel {{
            width: 55%;
            padding: 10px;
            border-right: 3px solid #ccc;
            overflow-y: scroll;
        }}

        .right-panel {{
            width: 45%;
            padding: 20px;
            overflow-y: scroll;
        }}

        .topic-block {{
            background: #fafafa;
            margin-bottom: 25px;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #ddd;
        }}

        h2 {{
            margin-top: 0;
        }}
    </style>

</head>
<body>

<div class="header">
    Dashboard updated: {timestamp}
</div>

<div class="main-container">

    <!-- Left: Upgraded BERTopic map -->
    <div class="left-panel">
        <h2>BERTopic Topic Map</h2>
        {topic_map_html}

        <h2>Top Words per Topic</h2>
        {barchart_html}
    </div>

    <!-- Right: GPT summaries -->
    <div class="right-panel">
        <h2>Topic Summaries (GPT)</h2>
"""

    # Insert summaries
    for tid in sorted(topic_summaries.keys()):
        s = topic_summaries[tid]

        html += f"""
        <div class="topic-block">
            <h3>Topic {tid}: {s['label']}</h3>
            <p>{s['description']}</p>

            <b>Subthemes:</b>
            <ul>
                {''.join(f"<li>{st}</li>" for st in s['subthemes'])}
            </ul>
        </div>
        """

    html += """
    </div>
</div>

</body>
</html>
"""

    # Save file
    out_file = os.path.join(OUTPUT_DIR, "index.html")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Dashboard created → {out_file}")


if __name__ == "__main__":
    generate_dashboard()
