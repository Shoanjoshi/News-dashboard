# ============================================
# 📄 generate_dashboard.py
# Version 6.0 – Theme layout upgrade + Topic×Theme heatmap + JSON-safe theme signals
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
# Save topic embeddings (for persistence)
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
# Load previous theme metrics
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
# SAVE theme metrics (Option A FIX — convert sets → lists)
# --------------------------------------------
def _save_theme_signals(theme_metrics):
    """Convert any sets to lists before saving JSON."""
    safe_metrics = {}

    for theme, metrics in theme_metrics.items():
        cleaned = {}
        for k, v in metrics.items():
            if isinstance(v, set):
                cleaned[k] = list(v)
            else:
                cleaned[k] = v
        safe_metrics[theme] = cleaned

    try:
        with open(THEME_SIGNALS_JSON, "w", encoding="utf-8") as f:
            json.dump(safe_metrics, f, indent=2)
    except Exception as e:
        print("❌ Error saving theme metrics:", e)


# --------------------------------------------
# Stretch plotly figure
# --------------------------------------------
def _stretch_figure(fig):
    fig.update_layout(autosize=True, margin=dict(l=5, r=5, t=40, b=5))
    return fig


# --------------------------------------------
# THEME SCATTER MAP
# --------------------------------------------
def _build_theme_map_html(theme_signals, total_docs):
    try:
        themes = list(theme_signals.keys())
        xs = [theme_signals[t]["topicality"] for t in themes]
        ys = [theme_signals[t]["centrality"] for t in themes]

        fig = go.Figure(
            data=[
                go.Scatter(
                    x=xs, y=ys, mode="markers+text",
                    text=themes, textposition="top center"
                )
            ]
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
# TOPIC ✕ THEME HEATMAP
# --------------------------------------------
def _build_heatmap(topic_summaries, theme_metrics, docs, topic_model):
    """Creates heatmap: rows=topic titles, columns=themes, values=article counts."""
    try:
        ordered_topics = sorted(topic_summaries.keys())

        row_labels = [topic_summaries[t]["title"] for t in ordered_topics]
        col_labels = THEMES + ["Others"]

        heat = np.zeros((len(ordered_topics), len(col_labels)), dtype=int)

        # Correct topic index extraction for BERTopic
        topics = topic_model.topics_

        for article_id, topic_id in enumerate(topics):
            if topic_id not in topic_summaries:
                continue

            row = ordered_topics.index(topic_id)

            for col, theme in enumerate(col_labels):
                article_list = theme_metrics[theme].get("articles_raw", [])

                # JSON fix means these are lists, so this works
                if article_id in article_list:
                    heat[row, col] += 1

        fig = go.Figure(
            data=go.Heatmap(
                z=heat,
                x=col_labels,
                y=row_labels,
                colorscale="Blues",
                showscale=True,
                colorbar=dict(lenmode="fraction", len=0.8),
            )
        )

        fig.update_layout(title="Topic × Theme Volume Heatmap")

        return fig.to_html(full_html=False)

    except Exception as e:
        print("⚠️ Heatmap error:", e)
        return "<p>Heatmap unavailable.</p>"


# ============================================
# 🚀 MAIN DASHBOARD FUNCTION
# ============================================
def generate_dashboard():
    print("🚀 Generating dashboard...")

    docs, topic_summaries, topic_model, embeddings, theme_scores = generate_topic_results()
    total_docs = len(docs)

    if not docs:
        print("⚠️ No docs.")
        return

    _save_topic_embeddings(embeddings, topic_summaries)

    # Load previous theme metrics
    prev = _load_previous_theme_signals()

    # Build today's metrics baseline
    theme_metrics = {
        theme: {
            "volume": info.get("volume", 0),
            "centrality": info.get("centrality", 0.0),
            "articles_raw": info.get("articles_raw", []),  # used for heatmap
        }
        for theme, info in theme_scores.items()
    }

    # Compute topicality = % change
    for theme, m in theme_metrics.items():
        prev_vol = prev.get(theme, {}).get("volume", 0)
        m["topicality"] = (m["volume"] - prev_vol) / prev_vol if prev_vol else 0

    # Rankings
    for metric in ["centrality", "topicality"]:
        ordered = sorted(theme_metrics.items(), key=lambda x: -x[1][metric])
        for r, (theme, _) in enumerate(ordered, start=1):
            theme_metrics[theme][f"{metric}_rank"] = r

    # Format theme_signals for template
    theme_signals = {
        t: {
            "centrality": round(m["centrality"], 3),
            "topicality": round(m["topicality"], 3),
            "centrality_rank": m["centrality_rank"],
            "topicality_rank": m["topicality_rank"],
            "prev_centrality": round(prev.get(t, {}).get("centrality", 0), 3)
                if prev.get(t) else None,
            "prev_topicality": prev.get(t, {}).get("topicality", None),
            "prev_centrality_rank": prev.get(t, {}).get("centrality_rank", None),
            "prev_topicality_rank": prev.get(t, {}).get("topicality_rank", None),
        }
        for t, m in theme_metrics.items()
    }

    # Save cleaned theme_metrics JSON
    _save_theme_signals(theme_metrics)

    # Build plots
    theme_map = _build_theme_map_html(theme_signals, total_docs)
    heatmap_html = _build_heatmap(topic_summaries, theme_metrics, docs, topic_model)
    topic_map_html = _stretch_figure(topic_model.visualize_topics()).to_html(full_html=False)

    # Summary list for template
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

    # Render HTML
    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("dashboard_template.html")

    html = template.render(
        topic_map=topic_map_html,
        theme_map=theme_map,
        heatmap=heatmap_html,
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
