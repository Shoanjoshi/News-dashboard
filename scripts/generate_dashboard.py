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
    # 1. Full pipeline (from Version 4)
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

    # Parse JSON outputs
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
    # 2. Metadata extraction for improved visualization
    # ------------------------------------------------------------
    doc_info = topic_model.get_document_info(texts)
    topic_info = topic_model.get_topic_info()

    valid_topics = topic_info[topic_info["Topic"] != -1]["Topic"].tolist()

    hover_texts = []
    custom_labels = []

    for tid in valid_topics:
        label = topic_summaries[tid]["label"]

        # Top words
        top_words = topic_model.get_topic(tid)
        top_words_text = ", ".join([w[0] for w in top_words[:5]]) if top_words else ""

        # Top headlines
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

        hover_texts.append(
            f"<b>{label}</b><br>"
            f"<b>Top words:</b> {top_words_text}<br>"
            f"<b>Headlines:</b> {headlines_text}"
        )

        custom_labels.append(label)

    # ------------------------------------------------------------
    # 3. Upgraded BERTopic visualization (Version 5.1 + Version 6)
    # ------------------------------------------------------------
    fig = topic_model.visualize_topics(
        custom_labels=True,
        width=1000,
        height=850
    )

    # ===== Modern color palette (Version 6) =====
    pastel_palette = [
        "#7DA1C4", "#A3C4BC", "#D7E3E7",
        "#C4A287", "#E8D7C1", "#9BA6B2",
        "#C3BABA", "#8E9AAF", "#B8CBD0",
    ]

    # assign colors cycling through pastel palette
    num_points = len(fig.data[0].x)
    fig.data[0].marker.color = [
        pastel_palette[i % len(pastel_palette)] for i in range(num_points)
    ]

    # ===== Bigger bubbles (executive look) =====
    fig.data[0].marker.size = [55] * num_points
    fig.data[0].marker.opacity = 0.88
    fig.data[0].marker.line.width = 1
    fig.data[0].marker.line.color = "#444"

    # ===== Hover tooltips =====
    fig.data[0].text = hover_texts
    fig.data[0].hovertemplate = "%{text}<extra></extra>"

    # ===== Add labels next to bubbles =====
    for i, label in enumerate(custom_labels):
        x = fig.data[0].x[i]
        y = fig.data[0].y[i]

        fig.add_annotation(
            x=x,
            y=y,
            text=f"<b>{label}</b>",
            showarrow=False,
            font=dict(size=15, color="#222"),
            xanchor="left",
            yanchor="middle"
        )

    # Convert map to HTML snippet
    topic_map_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

    # ------------------------------------------------------------
    # 4. Resized barchart (no horizontal scrolling)
    # ------------------------------------------------------------
    barchart_fig = topic_model.visualize_barchart(
        top_n_topics=NUM_TOPICS_FOR_LDA,
        width=750,
        height=500
    )

    barchart_fig.update_layout(
        margin=dict(l=20, r=20),
        font=dict(size=13),
    )

    barchart_html = barchart_fig.to_html(
        full_html=False,
        include_plotlyjs=False
    )

    # ------------------------------------------------------------
    # 5. Build Dashboard HTML (same layout as V4/V5)
    # ------------------------------------------------------------
    timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    html = f"""
<html>
<head>
    <title>Daily News Topic Dashboard</title>

    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', Arial, sans-serif;
            background-color: #fafafa;
        }}

        .header {{
            background: #E7EBEF;
            padding: 14px;
            font-size: 17px;
            font-weight: bold;
            border-bottom: 2px solid #d0d0d0;
        }}

        .main-container {{
            display: flex;
            height: calc(100vh - 60px);
        }}

        .left-panel {{
            width: 52%;
            padding: 10px;
            border-right: 3px solid #ccc;
            overflow-y: scroll;
        }}

        .right-panel {{
            width: 48%;
            padding: 20px;
            overflow-y: scroll;
        }}

        .topic-block {{
            background: #ffffff;
            margin-bottom: 20px;
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

    <div class="left-panel">
        <h2>Topic Distance Map</h2>
        {topic_map_html}

        <h2>Top Words (per Topic)</h2>
        {barchart_html}
    </div>

    <div class="right-panel">
        <h2>Topic Summaries (GPT)</h2>
"""

    # Write summaries
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

    out_file = os.path.join(OUTPUT_DIR, "index.html")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Dashboard created → {out_file}")


if __name__ == "__main__":
    generate_dashboard()
