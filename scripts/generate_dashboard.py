
# ============================================
# 📄 generate_dashboard.py — Topic Map Label Upgrade (T4 style)
# ============================================

import os
import json
from jinja2 import Environment, FileSystemLoader
import plotly.graph_objects as go
import numpy as np

from LDA_engine_with_BERTopic_v054 import generate_topic_results, THEMES

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
# Load prior theme metrics
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
def _save_theme_signals(theme_metrics):
    """Ensure data is JSON-serializable."""
    serializable = {
        k: {
            "volume": int(v.get("volume", 0)),
            "centrality": float(v.get("centrality", 0)),
            "topicality": float(v.get("topicality", 0)),
            "centrality_rank": v.get("centrality_rank"),
            "topicality_rank": v.get("topicality_rank"),
        }
        for k, v in theme_metrics.items()
    }

    with open(THEME_SIGNALS_JSON, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)


# --------------------------------------------
# Build Theme Scatter
# --------------------------------------------
def _build_theme_map_html(theme_signals, total_docs):
    themes = list(theme_signals.keys())
    xs = [theme_signals[t]["topicality"] for t in themes]
    ys = [theme_signals[t]["centrality"] for t in themes]

    fig = go.Figure(
        data=[
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers+text",
                text=themes,
                textposition="top center"
            )
        ]
    )

    fig.update_layout(
        title=f"Theme Distance Map – {total_docs} articles",
        xaxis_title="Topicality (% change)",
        yaxis_title="Centrality",
        autosize=True,
        height=550
    )

    return fig.to_html(full_html=False)


# --------------------------------------------
# Build Topic × Theme Heatmap
# --------------------------------------------
def _build_heatmap(topic_summaries, theme_metrics, docs, topic_model):
    try:
        ordered_topics = sorted(topic_summaries.keys())
        row_labels = [topic_summaries[t]["title"] for t in ordered_topics]
        col_labels = THEMES + ["Others"]

        heat = np.zeros((len(ordered_topics), len(col_labels)), dtype=int)

        # Article → Topic assignments
        topics = topic_model.topics_

        # theme_metrics now stores raw article IDs
        for row_index, topic_id in enumerate(ordered_topics):
            article_ids = [i for i, t in enumerate(topics) if t == topic_id]

            for aid in article_ids:
                for col, theme in enumerate(col_labels):
                    if aid in theme_metrics[theme]["articles_raw"]:
                        heat[row_index, col] += 1

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
            height=600,
        )

        return fig.to_html(full_html=False)

    except Exception as e:
        print("⚠️ Heatmap error:", e)
        return "<p>Heatmap unavailable.</p>"


# --------------------------------------------
# Build Topic Map with NEW “T4” Labels
# --------------------------------------------
def _build_topic_map_html(topic_model, topic_summaries):
    try:
        fig = topic_model.visualize_topics()

        topic_ids = sorted(topic_summaries.keys())
        labels = [f"T{tid}" for tid in topic_ids]

        fig.update_traces(
            text=labels,
            mode="markers+text",
            textposition="middle center",
            textfont=dict(size=14, color="white", family="Segoe UI"),
            marker=dict(size=22)
        )

        fig.update_layout(
            autosize=True,
            width=None,
            height=650,
            margin=dict(l=0, r=0, t=40, b=0)
        )

        return fig.to_html(full_html=False)

    except Exception as e:
        print("⚠️ Error generating topic map:", e)
        return "<p>No topic map available.</p>"


# ============================================
# 🚀 Main Dashboard Builder
# ============================================
def generate_dashboard():
    print("🚀 Generating dashboard...")

    docs, topic_summaries, topic_model, embeddings, theme_scores = generate_topic_results()
    total_docs = len(docs)

    if not docs:
        print("⚠️ No docs.")
        return

    _save_topic_embeddings(embeddings, topic_summaries)

    prev = _load_previous_theme_signals()

    # Build theme metrics
    theme_metrics = {}
    for theme, info in theme_scores.items():
        theme_metrics[theme] = {
            "volume": info["volume"],
            "centrality": info["centrality"],
            "articles_raw": info["articles_raw"],  # KEEP for heatmap
        }

    # Compute topicality (% change)
    for theme, m in theme_metrics.items():
        prev_vol = prev.get(theme, {}).get("volume", 0)
        m["topicality"] = (m["volume"] - prev_vol) / prev_vol if prev_vol else 0

    # Rankings
    for metric in ["centrality", "topicality"]:
        ordered = sorted(theme_metrics.items(), key=lambda x: -x[1][metric])
        for r, (theme, _) in enumerate(ordered, start=1):
            theme_metrics[theme][f"{metric}_rank"] = r

    # Build theme signals for HTML
    theme_signals = {
        theme: {
            "centrality": round(m["centrality"], 3),
            "topicality": round(m["topicality"], 3),
            "centrality_rank": m["centrality_rank"],
            "topicality_rank": m["topicality_rank"],
            "prev_centrality": round(prev.get(theme, {}).get("centrality", 0), 3)
                if prev.get(theme) else None,
            "prev_topicality": prev.get(theme, {}).get("topicality"),
            "prev_centrality_rank": prev.get(theme, {}).get("centrality_rank"),
            "prev_topicality_rank": prev.get(theme, {}).get("topicality_rank"),
        }
        for theme, m in theme_metrics.items()
    }

    _save_theme_signals(theme_metrics)

    topic_map_html = _build_topic_map_html(topic_model, topic_summaries)
    theme_map_html = _build_theme_map_html(theme_signals, total_docs)
    heatmap_html = _build_heatmap(topic_summaries, theme_metrics, docs, topic_model)

    summaries_list = [
        {
            "topic_id": tid,
            "title": info["title"],
            "summary": info["summary"].replace("\n", "<br>"),
            "article_count": info.get("article_count"),
            "is_new": info.get("status") == "NEW",
            "is_persistent": info.get("status") == "PERSISTENT",
        }
        for tid, info in topic_summaries.items()
    ]

    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("dashboard_template.html")

    html = template.render(
        topic_map=topic_map_html,
        theme_map=theme_map_html,
        heatmap=heatmap_html,
        theme_signals=theme_signals,
        summaries=summaries_list,
        run_date=os.getenv("RUN_DATE", "Today"),
        total_docs=total_docs,
    )

    with open(os.path.join(OUTPUT_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)

    print("🎉 Dashboard updated.")


if __name__ == "__main__":
    generate_dashboard()
