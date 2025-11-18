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
    # 1. Full pipeline (same as before)
    # ------------------------------------------------------------
    print("Fetching articles...")
    articles = fetch_articles(MAX_ARTICLES_TO_FETCH)
    texts = [a["text"] for a in articles]

    print("Running BERTopic...")
    topic_model, topics, probs = build_bertopic_model(texts)

    print("Generating GPT topic summaries...")
    raw_summaries = interpret_topics(topic_model, texts)

    topic_summaries = {}
    for tid, js in raw_summaries.items():
        try:
            topic_summaries[tid] = json.loads(js)
        except:
            topic_summaries[tid] = {
                "topic_id": tid,
                "label": f"Topic {tid}",
                "description": js,
                "subthemes": []
            }

    # ------------------------------------------------------------
    # 2. Extract metadata
    # ------------------------------------------------------------
    doc_info = topic_model.get_document_info(texts)
    topic_info = topic_model.get_topic_info()

    valid_topics = [
        t for t in topic_info["Topic"].tolist()
        if t != -1
    ]

    hover_texts = []
    custom_labels = []

    for tid in valid_topics:

        label = topic_summaries[tid]["label"]

        top_words = topic_model.get_topic(tid)
        top_word_str = ", ".join([w[0] for w in top_words[:5]]) if top_words else ""

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

        headline_str = "; ".join(headlines) if headlines else "No headline"

        hover_texts.append(
            f"<b>{label}</b><br>"
            f"<b>Top words:</b> {top_word_str}<br>"
            f"<b>Headlines:</b> {headline_str}"
        )

        custom_labels.append(label)

    # ------------------------------------------------------------
    # 3. Fix A — Safe fallback if no topic embeddings available
    # ------------------------------------------------------------
    if (
        topic_model.topic_embeddings_ is None
        or len(topic_model.topic_embeddings_) < 2
    ):
        print("WARNING: Not enough topics to visualize — using fallback.")
        topic_map_html = "<p><i>Not enough distinct topics today to generate a map.</i></p>"

    else:
        print("Generating upgraded topic map...")

        fig = topic_model.visualize_topics(
            custom_labels=True,
            width=1000,
            height=850
        )

        pastel_palette = [
            "#7DA1C4", "#A3C4BC", "#D7E3E7",
            "#C4A287", "#E8D7C1", "#9BA6B2",
            "#C3BABA", "#8E9AAF", "#B8CBD0",
        ]

        num_points = len(fig.data[0].x)
        fig.data[0].marker.color = [
            pastel_palette[i % len(pastel_palette)]
            for i in range(num_points)
        ]

        fig.data[0].marker.size = [55] * num_points
        fig.data[0].marker.opacity = 0.88
        fig.data[0].marker.line.width = 1
        fig.data[0].marker.line.color = "#444"

        fig.data[0].text = hover_texts
        fig.data[0].hovertemplate = "%{text}<extra></extra>"

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

        topic_map_html = fig.to_html(
            full_html=False,
            include_plotlyjs="cdn"
        )

    # ------------------------------------------------------------
    # 4. Resized barchart
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
    # 5. Dashboard HTML
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

    out_path = os.path.join(OUTPUT_DIR, "index.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Dashboard created → {out_path}")


if __name__ == "__main__":
    generate_dashboard()
