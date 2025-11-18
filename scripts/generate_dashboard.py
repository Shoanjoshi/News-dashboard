import os
import json
import pandas as pd
from datetime import datetime
from jinja2 import Environment, FileSystemLoader

from LDA_engine_with_BERTopic_v04 import (
    fetch_articles,
    build_bertopic_model,
    interpret_topics,
)

# Output directory for GitHub Pages
OUTPUT_DIR = "dashboard"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_dashboard():

    print("Fetching articles...")
    articles = fetch_articles()
    texts = [a["text"] for a in articles]
    titles = [a["title"] for a in articles]
    urls = [a["url"] for a in articles]

    if len(texts) == 0:
        print("No articles found. Creating fallback dashboard.")
        with open(os.path.join(OUTPUT_DIR, "index.html"), "w") as f:
            f.write("<h1>No articles found today.</h1>")
        return

    print("Running BERTopic...")
    topic_model, topics, probs = build_bertopic_model(texts)

    # Build document-level data
    doc_df = pd.DataFrame({
        "title": titles,
        "url": urls,
        "text": texts,
        "topic": topics,
    })

    # --------------------------------------
    # Determine valid topics (excluding -1)
    # --------------------------------------
    topic_info = topic_model.get_topic_info()
    valid_topics = [
        t for t in topic_info["Topic"].tolist()
        if t != -1
    ]

    print("Valid topics found:", valid_topics)

    # --------------------------------------
    # GPT Summaries (with fallback)
    # --------------------------------------
    if len(valid_topics) < 1:
        print("WARNING: No valid topics → creating fallback summary section.")
        topic_summaries = {
            "No Topics": {
                "label": "No topics identified",
                "description": "There were not enough distinct articles today for topic discovery.",
                "subthemes": []
            }
        }
    else:
        print("Generating GPT topic summaries...")
        raw_summaries = interpret_topics(topic_model, texts)
        topic_summaries = {}

        # Parse JSON safely
        for tid, content in raw_summaries.items():
            try:
                parsed = json.loads(content)
            except Exception:
                parsed = {
                    "label": f"Topic {tid}",
                    "description": "Summary generation failed.",
                    "subthemes": []
                }
            topic_summaries[tid] = parsed

    # -------------------------------------------------------------------
    # TOPIC MAP (FALLBACK if < 2 topics OR BERTopic visualization fails)
    # -------------------------------------------------------------------
    print("Rendering topic map...")
    if len(valid_topics) < 2:
        print("WARNING: Not enough topics to visualize — using fallback.")
        topic_map_html = "<p><i>Topic map unavailable (less than 2 topics).</i></p>"
    else:
        try:
            fig = topic_model.visualize_topics(
                custom_labels=True,
                width=1100,
                height=800,
            )
            topic_map_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
        except Exception as e:
            print(f"WARNING: Topic map render failed → {e}")
            topic_map_html = "<p><i>Topic map could not be generated.</i></p>"

    # -------------------------------------------------------------------
    # BAR CHART (FALLBACK if < 1 topic OR BERTopic barchart fails)
    # -------------------------------------------------------------------
    print("Rendering barchart...")
    if len(valid_topics) < 1:
        print("WARNING: Not enough topics to build barchart — fallback.")
        barchart_html = "<p><i>No topic-term bar chart available today.</i></p>"
    else:
        try:
            barchart_fig = topic_model.visualize_barchart(
                top_n_topics=min(12, len(valid_topics)),
                width=1100,
                height=600,
            )
            barchart_html = barchart_fig.to_html(full_html=False, include_plotlyjs="cdn")
        except Exception as e:
            print(f"WARNING: Barchart failed → {e}")
            barchart_html = "<p><i>Bar chart could not be generated.</i></p>"

    # --------------------------------------
    # Build final dashboard HTML via Jinja2
    # --------------------------------------
    print("Building final HTML...")
    env = Environment(loader=FileSystemLoader("templates/"))
    template = env.get_template("dashboard_template.html")

    html = template.render(
        generated_on=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        topic_map_html=topic_map_html,
        barchart_html=barchart_html,
        topic_summaries=topic_summaries,
    )

    output_path = os.path.join(OUTPUT_DIR, "index.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print("Dashboard generated:", output_path)


if __name__ == "__main__":
    generate_dashboard()
