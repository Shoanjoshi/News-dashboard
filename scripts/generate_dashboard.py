# ============================================
# 📄 generate_dashboard.py
# Version 5.4 – Dashboard + topic persistence
# ============================================

import os
import json
import numpy as np
from jinja2 import Environment, FileSystemLoader
from sklearn.metrics.pairwise import cosine_similarity

from LDA_engine_with_BERTopic_v054 import generate_topic_results

OUTPUT_DIR = "dashboard"
DATA_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

YESTERDAY_FILE = os.path.join(DATA_DIR, "yesterday_topics.json")
PERSISTENCE_THRESHOLD = 0.75  # cosine similarity threshold


def load_yesterday_topics():
    if not os.path.exists(YESTERDAY_FILE):
        print("ℹ️ No yesterday_topics.json found (first run or reset).")
        return []

    try:
        with open(YESTERDAY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("topics", [])
    except Exception as e:
        print(f"⚠️ Failed to load yesterday topics: {e}")
        return []


def save_today_topics(topic_summaries, topic_embeddings):
    topics_to_save = []
    for topic_id, meta in topic_summaries.items():
        emb = topic_embeddings.get(topic_id)
        if emb is None:
            continue
        topics_to_save.append(
            {
                "topic_id": int(topic_id),
                "title": meta.get("title", f"Topic {topic_id}"),
                "summary": meta.get("summary", ""),
                "embedding": emb,
            }
        )

    payload = {"topics": topics_to_save}
    with open(YESTERDAY_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    print(f"💾 Saved {len(topics_to_save)} topics to {YESTERDAY_FILE}")


def compute_persistence(today_summaries, today_embeddings, yesterday_topics):
    """
    today_summaries: dict[topic_id] -> {"title", "summary"}
    today_embeddings: dict[topic_id] -> [float,...]
    yesterday_topics: list of dicts with keys: title, summary, embedding
    """
    if not yesterday_topics:
        # First run: everything is new
        return {tid: "NEW" for tid in today_summaries.keys()}

    prev_embs = np.array([t["embedding"] for t in yesterday_topics], dtype=float)
    persistence_labels = {}

    for topic_id, meta in today_summaries.items():
        emb_today = np.array(today_embeddings.get(topic_id), dtype=float).reshape(1, -1)
        sims = cosine_similarity(emb_today, prev_embs)[0]
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        best_prev = yesterday_topics[best_idx]

        if best_sim >= PERSISTENCE_THRESHOLD:
            label = f"Matched to: {best_prev['title']} ({best_sim:.2f})"
        else:
            label = "NEW"

        persistence_labels[topic_id] = label

    return persistence_labels


def generate_dashboard():
    print("🚀 Generating dashboard...")
    docs, topic_summaries, topic_model, topic_embeddings = generate_topic_results()

    # Fallback if no docs/model
    if not docs or not topic_model or not topic_summaries:
        fallback_html = "<h3>⚠️ Not enough data to generate dashboard today.</h3>"
        with open(os.path.join(OUTPUT_DIR, "index.html"), "w", encoding="utf-8") as f:
            f.write(fallback_html)
        return

    # --- Compute persistence ---
    yesterday_topics = load_yesterday_topics()
    persistence_map = compute_persistence(topic_summaries, topic_embeddings, yesterday_topics)

    # --- Build visualizations (topic map & barchart) ---
    try:
        fig_topics = topic_model.visualize_topics(width=600, height=650)
        html_topic_map = fig_topics.to_html(full_html=False)
    except Exception as e:
        print(f"⚠️ Failed to build topic map: {e}")
        html_topic_map = "<p>No topic map available.</p>"

    try:
        fig_barchart = topic_model.visualize_barchart(top_n_topics=5)
        html_barchart = fig_barchart.to_html(full_html=False)
    except Exception as e:
        print(f"⚠️ Failed to build barchart: {e}")
        html_barchart = "<p>No keyword barchart available.</p>"

    # --- Table data for template ---
    summary_list = []
    for topic_id, meta in topic_summaries.items():
        summary_list.append(
            {
                "topic_id": topic_id,
                "title": meta.get("title", f"Topic {topic_id}"),
                "summary": meta.get("summary", ""),
                "persistence": persistence_map.get(topic_id, "NEW"),
            }
        )

    # --- Render HTML ---
    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("dashboard_template.html")
    rendered_html = template.render(
        topic_map=html_topic_map,
        barchart=html_barchart,
        summaries=summary_list,
    )

    out_path = os.path.join(OUTPUT_DIR, "index.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(rendered_html)

    # --- Save today's topics for tomorrow's run ---
    save_today_topics(topic_summaries, topic_embeddings)

    print(f"🎉 Dashboard generated → {out_path}")


if __name__ == "__main__":
    generate_dashboard()
