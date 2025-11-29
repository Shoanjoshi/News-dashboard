# ============================================
# 📄 generate_dashboard.py
# Version 5.8 – Updated metrics, centrality, and article counts
# ============================================

import os
import json
from jinja2 import Environment, FileSystemLoader
import plotly.graph_objects as go
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from LDA_engine_with_BERTopic_v054 import generate_topic_results

# --------------------------------------------
# 1️⃣ OpenAI Key Validation
# --------------------------------------------
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("⚠️ OPENAI_API_KEY not found. Add it as a GitHub Secret.")

# --------------------------------------------
# 2️⃣ Output Directory
# --------------------------------------------
OUTPUT_DIR = "dashboard"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOPIC_PERSISTENCE_JSON = os.path.join(OUTPUT_DIR, "yesterday_topics.json")
THEME_SIGNALS_JSON = os.path.join(OUTPUT_DIR, "yesterday_theme_signals.json")

# Centrality threshold (similarity)
CENTRALITY_SIMILARITY_THRESHOLD = 0.6


# --------------------------------------------
# Helper functions
# --------------------------------------------
def _save_topic_embeddings(embeddings, topic_summaries):
    try:
        with open(TOPIC_PERSISTENCE_JSON, "w", encoding="utf-8") as f:
            json.dump(
                {
                    str(k): {
                        "embedding": embeddings[k],
                        "title": topic_summaries[k].get("title", "")
                    }
                    for k in embeddings
                },
                f, indent=2)
        print(f"📁 Saved topic persistence file → {TOPIC_PERSISTENCE_JSON}")
    except Exception as e:
        print(f"❌ Error saving topic JSON: {e}")


def _load_previous_theme_signals():
    if not os.path.exists(THEME_SIGNALS_JSON):
        print("🟡 No previous theme signals found – first-run conditions.")
        return {}
    try:
        with open(THEME_SIGNALS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Error loading prior theme data: {e}")
        return {}


def _save_theme_signals(theme_metrics):
    try:
        with open(THEME_SIGNALS_JSON, "w", encoding="utf-8") as f:
            json.dump(theme_metrics, f, indent=2)
        print(f"📁 Saved theme metrics → {THEME_SIGNALS_JSON}")
    except Exception as e:
        print(f"❌ Error saving theme metrics: {e}")


def _stretch_figure(fig):
    fig.update_layout(
        autosize=True,
        height=None,
        width=None,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def _build_theme_map_html(theme_signals, total_docs):
    if not theme_signals:
        return "<p>No theme visualization available.</p>"

    try:
        themes = list(theme_signals.keys())
        xs = [theme_signals[t]["topicality"] for t in themes]
        ys = [theme_signals[t]["centrality"] for t in themes]

        fig = go.Figure(data=[
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers+text",
                text=themes,
                textposition="top center",
            )
        ])

        fig.update_layout(
            title=f"Theme Distance Map (Centrality vs Δ Volume) – {total_docs} articles",
            xaxis_title="Topicality (Δ news volume vs prior day)",
            yaxis_title="Centrality (semantic theme overlap)",
            autosize=True,
        )

        return fig.to_html(full_html=False)

    except Exception as e:
        print(f"⚠️ Error generating theme map: {e}")
        return "<p>No theme visualization available.</p>"


# ------------------------------------------------
# 🚀 Generate Dashboard
# ------------------------------------------------
def generate_dashboard():
    print("🚀 Generating upgraded dashboard...")

    docs, topic_summaries, topic_model, embeddings, theme_scores = generate_topic_results()
    total_docs = len(docs)

    if not docs or not topic_model:
        output_path = os.path.join(OUTPUT_DIR, "index.html")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("<h3>No sufficient data to generate dashboard.</h3>")
        print("⚠️ Dashboard fallback created.")
        return

    _save_topic_embeddings(embeddings, topic_summaries)

    try:
        fig_topics = topic_model.visualize_topics()
        fig_topics = _stretch_figure(fig_topics)
        html_topic_map = fig_topics.to_html(full_html=False)
    except:
        html_topic_map = "<p>No topic map available.</p>"

    # ️⃣ Improved theme logic
    prev_data = _load_previous_theme_signals()
    theme_metrics = {}

    # 1. Start with predefined categories
    for theme, info in theme_scores.items():
        theme_metrics[theme] = {
            "volume": int(info.get("volume", 0)),
            "centrality": float(info.get("centrality", 0.0)),
        }

    # 2. Add “Others” theme if missing volume
    total_theme_volume = sum([v["volume"] for v in theme_metrics.values()])
    if total_docs > total_theme_volume:
        theme_metrics["Others"] = {
            "volume": total_docs - total_theme_volume,
            "centrality": 0.0  # No similarity data yet
        }

    # 3. Centrality based on embedding proximity
    themes = list(theme_metrics.keys())
    vectors = np.array([
        embeddings.get(theme, [0]*len(list(embeddings.values())[0])) for theme in themes
    ])

    for i, theme in enumerate(themes):
        sim_scores = cosine_similarity([vectors[i]], vectors)[0]
        centrality = sum(sim_scores > CENTRALITY_SIMILARITY_THRESHOLD) - 1
        theme_metrics[theme]["centrality"] = centrality

    # 4. Topicality remains Δ volume (your request)
    for theme in theme_metrics:
        prev_volume = prev_data.get(theme, {}).get("volume", 0)
        theme_metrics[theme]["topicality"] = theme_metrics[theme]["volume"] - prev_volume

    # 5. Rank ordering
    for metric in ["centrality", "topicality"]:
        ranked = sorted(theme_metrics.items(), key=lambda x: -x[1][metric])
        for rank, (theme, _) in enumerate(ranked, start=1):
            theme_metrics[theme][f"{metric}_rank"] = rank

    # 6. Round values correctly (action 1)
    theme_signals = {
        theme: {
            "centrality": round(m["centrality"], 3),
            "topicality": round(m["topicality"], 3),
            "centrality_rank": m["centrality_rank"],
            "topicality_rank": m["topicality_rank"],
            "volume": m["volume"]
        }
        for theme, m in theme_metrics.items()
    }

    _save_theme_signals(theme_signals)

    # 7. Generate Theme Map
    html_theme_map = _build_theme_map_html(theme_signals, total_docs)

    # 8. Prepare Summary List
    summary_list = [
        {
            "topic_id": k,
            "title": v["title"],
            "summary": v["summary"].replace("\n", "<br>"),
            "is_new": v.get("status") == "NEW",
            "is_persistent": v.get("status") == "PERSISTENT",
        }
        for k, v in topic_summaries.items()
    ]

    # 9. Render Dashboard
    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("dashboard_template.html")
    rendered_html = template.render(
        topic_map=html_topic_map,
        theme_map=html_theme_map,
        theme_signals=theme_signals,
        summaries=summary_list,
        run_date=os.getenv("RUN_DATE", "Today"),
        total_docs=total_docs,
    )

    output_path = os.path.join(OUTPUT_DIR, "index.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(rendered_html)
    print("🎉 Dashboard updated successfully!")


if __name__ == "__main__":
    generate_dashboard()
