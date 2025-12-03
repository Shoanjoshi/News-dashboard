# ============================================
# 📄 generate_dashboard.py
# Version 6.0 – Stable Output Path + Theme Features
# ============================================

import os
import json
import numpy as np
from jinja2 import Environment, FileSystemLoader
import plotly.graph_objects as go

from LDA_engine_with_BERTopic_v054 import generate_topic_results, THEMES

# --------------------------------------------
# Validate OpenAI Key
# --------------------------------------------
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("⚠️ OPENAI_API_KEY missing.")

# --------------------------------------------
# Output Directory (FIXED to repo root)
# --------------------------------------------
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dashboard"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOPIC_PERSISTENCE_JSON = os.path.join(OUTPUT_DIR, "yesterday_topics.json")
THEME_SIGNALS_JSON = os.path.join(OUTPUT_DIR, "yesterday_theme_signals.json")


# --------------------------------------------
# Save topic embeddings
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
                f,
                indent=2,
            )
        print("📁 Saved topic persistence file.")
    except Exception as e:
        print("❌ Error saving topic JSON:", e)


# --------------------------------------------
# Load previous theme signals
# --------------------------------------------
def _load_previous_theme_signals():
    if not os.path.exists(THEME_SIGNALS_JSON):
        print("🟡 No previous theme signals found.")
        return {}
    try:
        with open(THEME_SIGNALS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}


# --------------------------------------------
# Save today's theme signals
# --------------------------------------------
def _save_theme_signals(theme_metrics):
    try:
        with open(THEME_SIGNALS_JSON, "w", encoding="utf-8") as f:
            json.dump(theme_metrics, f, indent=2)
    except Exception as e:
        print("❌ Error saving theme metrics:", e)


# --------------------------------------------
# Stretch plotly figure
# --------------------------------------------
def _stretch_figure(fig):
    fig.update_layout(
        autosize=True,
        margin=dict(l=5, r=5, t=40, b=5),
    )
    return fig


# --------------------------------------------
# Theme scatter plot
# --------------------------------------------
def _build_theme_map_html(theme_signals, total_docs):
    try:
        themes = list(theme_signals.keys())
        xs = [theme_signals[t]["topicality"] for t in themes]
        ys = [theme_signals[t]["centrality"] for t in themes]

        fig = go.Figure(
            data=[go.Scatter(
                x=xs, y=ys,
                mode="markers+text",
                text=themes,
                textposition="top center"
            )]
        )

        fig.update_layout(
            title=f"Theme Distance Map – {total_docs} articles",
            xaxis_title="Topicality (% change)",
            yaxis_title="Centrality",
            autosize=True,
        )

        return fig.to_html(full_html=False)

    except Exception as e:
        print("⚠️ Error generating theme scatter:", e)
        return "<p>Theme plot unavailable.</p>"


# --------------------------------------------
# Topic × Theme Heatmap
# --------------------------------------------
def _build_heatmap(topic_summaries, theme_metrics, docs, topic_model):
    try:
        ordered_topics = sorted(topic_summaries.keys())
        row_labels = [topic_summaries[t]["title"] for t in ordered_topics]
        col_labels = THEMES + ["Others"]

        heat = np.zeros((len(ordered_topics), len(col_labels)), dtype=int)

        # Correct mapping from BERTopic model
        topics = topic_model.topics_

        for doc_id, topic_id in enumerate(topics):
            if topic_id not in topic_summaries:
                continue
            row = ordered_topics.index(topic_id)
            for col, theme in enumerate(col_labels):
                if doc_id in theme_metrics[theme].get("articles_raw", []):
                    heat[row, col] += 1

        fig = go.Figure(
            data=go.Heatmap(
                z=heat,
                x=col_labels,
                y=row_labels,
                colorscale="Blues",
                showscale=True,
            )
        )
        fig.update_layout(
            title="Topic × Theme Volume Heatmap",
            xaxis_nticks=len(col_labels),
        )
        return fig.to_html(full_html=False)

    except Exception as e:
        print("⚠️ Heatmap error:", e)
        return "<p>Heatmap unavailable.</p>"


# ============================================
# 🚀 Main Dashboard Generation
# ============================================
def generate_dashboard():
    print("🚀 Generating dashboard...")

    docs, topic_summaries, topic_model, embeddings, theme_scores = generate_topic_results()
    total_docs = len(docs)

    if not docs:
        print("⚠️ No docs returned.")
        return

    _save_topic_embeddings(embeddings, topic_summaries)

    # Load previous
    prev_data = _load_previous_theme_signals()

    # Construct theme metrics
    theme_metrics = {}
    for theme, info in theme_scores.items():
        theme_metrics[theme] = {
            "volume": info["volume"],
            "centrality": info["centrality"]
        }

    # Topicality %
    for theme, m in theme_metrics.items():
        prev_vol = prev_data.get(theme, {}).get("volume", 0)
        m["topicality"] = ((m["volume"] - prev_vol) / prev_vol) if prev_vol else 0

    # Ranking
    for metric in ["centrality", "topicality"]:
        sorted_items = sorted(theme_metrics.items(), key=lambda x: -x[1][metric])
        for rank, (theme, _) in enumerate(sorted_items, start=1):
            theme_metrics[theme][f"{metric}_rank"] = rank

    # Prepare theme signals for template
    theme_signals = {
        theme: {
            "centrality": round(m["centrality"], 3),
            "topicality": round(m["topicality"], 3),
            "centrality_rank": m["centrality_rank"],
            "topicality_rank": m["topicality_rank"],
            "prev_centrality": round(prev_data.get(theme, {}).get("centrality", 0), 3)
                if prev_data.get(theme) else None,
            "prev_topicality": prev_data.get(theme, {}).get("topicality"),
            "prev_centrality_rank": prev_data.get(theme, {}).get("centrality_rank"),
            "prev_topicality_rank": prev_data.get(theme, {}).get("topicality_rank"),
        }
        for theme, m in theme_metrics.items()
    }

    # Save today’s theme metrics
    _save_theme_signals(theme_metrics)

    # Build plots
    theme_map_html = _build_theme_map_html(theme_signals, total_docs)
    heatmap_html = _build_heatmap(topic_summaries, theme_scores, docs, topic_model)

    # Topic summaries table
    summary_list = [
        {
            "topic_id": k,
            "title": v["title"],
            "summary": v["summary"].replace("\n", "<br>"),
            "article_count": v.get("article_count"),
            "is_new": v.get("status") == "NEW",
            "is_persistent": v.get("status") == "PERSISTENT",
        }
        for k, v in topic_summaries.items()
    ]

    # Render HTML template
    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("dashboard_template.html")

    html = template.render(
        topic_map=_stretch_figure(topic_model.visualize_topics()).to_html(full_html=False),
        theme_map=theme_map_html,
        heatmap=heatmap_html,
        theme_signals=theme_signals,
        summaries=summary_list,
        run_date=os.getenv("RUN_DATE", "Today"),
        total_docs=total_docs,
    )

    # Write dashboard output
    output_file = os.path.join(OUTPUT_DIR, "index.html")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)

    print("🎉 Dashboard updated successfully.")


if __name__ == "__main__":
    generate_dashboard()
