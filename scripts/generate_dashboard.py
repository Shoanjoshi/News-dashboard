# ============================================
# 📄 generate_dashboard.py
# Version 6.0 – Improved Topic Map (titles outside + size scaling)
# ============================================

import os
import json
import numpy as np
from jinja2 import Environment, FileSystemLoader
import plotly.graph_objects as go

from LDA_engine_with_BERTopic_v054 import (
    generate_topic_results,
    THEMES
)

# --------------------------------------------
# Validate OpenAI Key
# --------------------------------------------
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("⚠️ OPENAI_API_KEY missing.")

# --------------------------------------------
# Output Paths
# --------------------------------------------
OUTPUT_DIR = "dashboard"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOPIC_PERSISTENCE_JSON = os.path.join(OUTPUT_DIR, "yesterday_topics.json")
THEME_SIGNALS_JSON = os.path.join(OUTPUT_DIR, "yesterday_theme_signals.json")


# --------------------------------------------
# Save topic embeddings
# --------------------------------------------
def _save_topic_embeddings(embeddings, topic_summaries):
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


# --------------------------------------------
# Load prior theme signals
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
# Save today’s theme metrics
# --------------------------------------------
def _save_theme_signals(metrics):
    with open(THEME_SIGNALS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


# --------------------------------------------
# Improved Topic Map (titles outside + bubble scaling)
# --------------------------------------------
def _build_topic_map_html(topic_model, topic_summaries):
    try:
        fig = topic_model.visualize_topics()

        topic_ids = sorted(topic_summaries.keys())
        titles = [topic_summaries[t]["title"] for t in topic_ids]

        # Inside bubble text: T#
        tlabels = [f"T{t}" for t in topic_ids]

        # Bubble size scaling by article volume
        counts = np.array([
            topic_summaries[t]["article_count"]
            for t in topic_ids
        ])
        sizes = 20 + 60 * (counts - counts.min()) / (counts.max() - counts.min() + 1)

        # Update bubble markers + inside text
        fig.update_traces(
            mode="markers+text",
            text=tlabels,
            textposition="middle center",
            textfont=dict(size=12, color="white", family="Segoe UI"),
            marker=dict(size=sizes)
        )

        # Overlay topic titles outside the bubbles
        fig.add_trace(
            go.Scatter(
                x=fig.data[0].x,
                y=fig.data[0].y,
                mode="text",
                text=titles,
                textposition="top center",
                textfont=dict(size=13, color="black", family="Segoe UI"),
                hoverinfo="skip",
                showlegend=False
            )
        )

        # Layout fix → fills container, no squishing
        fig.update_layout(
            autosize=True,
            width=None,
            height=700,
            margin=dict(l=0, r=0, t=40, b=0)
        )

        return fig.to_html(full_html=False)

    except Exception as e:
        print("⚠️ Error building improved topic map:", e)
        return "<p>No topic map available.</p>"


# --------------------------------------------
# Scatter Map Builder (unchanged)
# --------------------------------------------
def _build_theme_map_html(theme_signals, total_docs):
    try:
        themes = list(theme_signals.keys())
        xs = [theme_signals[t]["topicality"] for t in themes]
        ys = [theme_signals[t]["centrality"] for t in themes]

        fig = go.Figure(
            data=[go.Scatter(
                x=xs,
                y=ys,
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
        print("⚠️ Error in theme scatter:", e)
        return "<p>Error.</p>"


# --------------------------------------------
# Heatmap (unchanged)
# --------------------------------------------
def _build_heatmap(topic_summaries, theme_metrics, docs, topic_model):
    try:
        ordered_topics = sorted(topic_summaries.keys())
        row_labels = [topic_summaries[t]["title"] for t in ordered_topics]
        col_labels = list(THEMES) + ["Others"]

        heat = np.zeros((len(ordered_topics), len(col_labels)), dtype=int)

        topics = topic_model.topics_

        for article_id, topic_id in enumerate(topics):
            if topic_id not in topic_summaries:
                continue

            row = ordered_topics.index(topic_id)

            for col, theme in enumerate(col_labels):
                if article_id in theme_metrics[theme].get("articles_raw", []):
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
# 🚀 Main Dashboard Function
# ============================================
def generate_dashboard():
    print("🚀 Generating dashboard...")

    docs, topic_summaries, topic_model, embeddings, theme_scores = generate_topic_results()
    total_docs = len(docs)

    _save_topic_embeddings(embeddings, topic_summaries)

    prev = _load_previous_theme_signals()
    theme_metrics = {}

    for theme, info in theme_scores.items():
        theme_metrics[theme] = {
            "volume": info["volume"],
            "centrality": info["centrality"],
        }

    # Topicality = % change
    for t, m in theme_metrics.items():
        p = prev.get(t, {}).get("volume", 0)
        m["topicality"] = (m["volume"] - p) / p if p else 0

    # Ranking
    for metric in ["centrality", "topicality"]:
        ranked = sorted(theme_metrics.items(), key=lambda x: -x[1][metric])
        for rank, (theme, _) in enumerate(ranked, 1):
            theme_metrics[theme][f"{metric}_rank"] = rank

    # Prepare template-friendly structure
    theme_signals = {
        t: {
            "centrality": round(m["centrality"], 3),
            "topicality": round(m["topicality"], 3),
            "centrality_rank": m["centrality_rank"],
            "topicality_rank": m["topicality_rank"],
            "prev_centrality": round(prev.get(t, {}).get("centrality", 0), 3)
                if prev.get(t) else None,
            "prev_topicality": prev.get(t, {}).get("topicality"),
            "prev_centrality_rank": prev.get(t, {}).get("centrality_rank"),
            "prev_topicality_rank": prev.get(t, {}).get("topicality_rank"),
        }
        for t, m in theme_metrics.items()
    }

    _save_theme_signals(theme_metrics)

    # Build plots
    html_topic_map = _build_topic_map_html(topic_model, topic_summaries)
    html_theme_map = _build_theme_map_html(theme_signals, total_docs)
    html_heatmap = _build_heatmap(topic_summaries, theme_scores, docs, topic_model)

    # Prepare summaries
    summary_list = [
        {
            "topic_id": t,
            "title": v["title"],
            "summary": v["summary"].replace("\n", "<br>"),
            "article_count": v["article_count"],
            "is_new": v.get("status") == "NEW",
            "is_persistent": v.get("status") == "PERSISTENT",
        }
        for t, v in topic_summaries.items()
    ]

    # Render Template
    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("dashboard_template.html")

    html = template.render(
        topic_map=html_topic_map,
        theme_map=html_theme_map,
        heatmap=html_heatmap,
        theme_signals=theme_signals,
        summaries=summary_list,
        run_date=os.getenv("RUN_DATE", "Today"),
        total_docs=total_docs,
    )

    with open(os.path.join(OUTPUT_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)

    print("🎉 Dashboard updated.")


if __name__ == "__main__":
    generate_dashboard()

