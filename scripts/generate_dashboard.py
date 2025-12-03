# ============================================
# 📄 generate_dashboard.py
# Version 6.0 – Hero topic styling + stable layout
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
                        "title": topic_summaries[k].get("title", ""),
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
    except Exception as e:
        print("⚠️ Error loading previous theme signals:", e)
        return {}


# --------------------------------------------
# Save today’s theme metrics (JSON-serialisable only)
# --------------------------------------------
def _save_theme_signals(theme_metrics):
    # Persist only scalar fields (no article lists)
    clean = {
        theme: {
            "volume": int(m.get("volume", 0)),
            "centrality": float(m.get("centrality", 0.0)),
            "topicality": float(m.get("topicality", 0.0)),
            "centrality_rank": int(m.get("centrality_rank", 0)),
            "topicality_rank": int(m.get("topicality_rank", 0)),
        }
        for theme, m in theme_metrics.items()
    }
    try:
        with open(THEME_SIGNALS_JSON, "w", encoding="utf-8") as f:
            json.dump(clean, f, indent=2)
    except Exception as e:
        print("❌ Error saving theme metrics:", e)


# --------------------------------------------
# Theme Scatter plot
# --------------------------------------------
def _build_theme_map_html(theme_signals, total_docs):
    try:
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
                    textposition="top center",
                )
            ]
        )

        fig.update_layout(
            title=f"Theme Distance Map – {total_docs} articles",
            xaxis_title="Topicality (% change)",
            yaxis_title="Centrality",
            autosize=True,
            margin=dict(l=10, r=10, t=40, b=10),
        )

        return fig.to_html(full_html=False)

    except Exception as e:
        print("⚠️ Error in theme scatter:", e)
        return "<p>Error.</p>"


# --------------------------------------------
# Topic × Theme Heatmap
# --------------------------------------------
def _build_heatmap(topic_summaries, theme_metrics, topic_model):
    """Rows = topic title; Cols = themes; Values = article counts."""
    try:
        if topic_model is None or not topic_summaries:
            return "<p>Heatmap unavailable.</p>"

        ordered_topic_ids = sorted(topic_summaries.keys())
        row_labels = [
            f"T{tid}: {topic_summaries[tid].get('title','')}" for tid in ordered_topic_ids
        ]
        col_labels = THEMES + ["Others"]

        heat = np.zeros((len(ordered_topic_ids), len(col_labels)), dtype=int)

        # BERTopic keeps article→topic assignment here
        article_topics = getattr(topic_model, "topics_", None)
        if article_topics is None:
            return "<p>Heatmap unavailable.</p>"

        # Make theme→article sets
        theme_article_sets = {
            theme: set(info.get("articles_raw", [])) for theme, info in theme_metrics.items()
        }

        # Build matrix
        for article_id, topic_id in enumerate(article_topics):
            if topic_id not in topic_summaries:
                continue
            row = ordered_topic_ids.index(topic_id)
            for col, theme in enumerate(col_labels):
                if article_id in theme_article_sets.get(theme, set()):
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

        fig.update_layout(
            title="Topic × Theme Volume Heatmap",
            xaxis_nticks=len(col_labels),
            autosize=True,
            margin=dict(l=10, r=10, t=40, b=10),
        )

        return fig.to_html(full_html=False)

    except Exception as e:
        print("⚠️ Heatmap error:", e)
        return "<p>Heatmap unavailable.</p>"


# --------------------------------------------
# Topic Map with hero topic styling
# --------------------------------------------
def _build_topic_map_html(topic_model, topic_summaries, hero_topic_ids):
    """Decorate BERTopic intertopic map with:
       - bubble size ~ article_count
       - hero topics (top 5) in light brown with title labels
       - others labeled T<ID>
    """
    if topic_model is None:
        return "<p>No topic map available.</p>"

    try:
        fig = topic_model.visualize_topics()

        # Find the main scatter trace
        scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
        if not scatter_traces:
            return fig.to_html(full_html=False)

        main_trace = scatter_traces[0]

        # Extract topic IDs from customdata (first element in each row)
        topic_ids = []
        custom = getattr(main_trace, "customdata", None)
        if custom is None:
            # Fallback: use numeric indices
            topic_ids = list(topic_summaries.keys())
        else:
            for cd in custom:
                if isinstance(cd, (list, tuple)) and len(cd) >= 1:
                    topic_ids.append(cd[0])
                else:
                    topic_ids.append(cd)

        # Make sure lengths align
        n_points = len(main_trace.x)
        if len(topic_ids) != n_points:
            # As a fallback, truncate/pad from sorted topic IDs
            sorted_ids = sorted(topic_summaries.keys())
            topic_ids = (sorted_ids + sorted_ids[-1:] * n_points)[:n_points]

        sizes = []
        texts = []
        colors = []

        HERO_COLOR = "#c98f4a"          # light brown
        NORMAL_COLOR = "rgba(120,144,156,0.85)"

        for tid in topic_ids:
            meta = topic_summaries.get(tid, {})
            count = meta.get("article_count", 1)
            base_size = 22
            size = base_size + 4 * (count ** 0.5)
            sizes.append(size)

            if tid in hero_topic_ids:
                title = meta.get("title", "")
                label = f"T{tid} – {title}"
                colors.append(HERO_COLOR)
            else:
                label = f"T{tid}"
                colors.append(NORMAL_COLOR)

            texts.append(label)

        main_trace.marker.size = sizes
        main_trace.marker.color = colors
        main_trace.marker.line = dict(width=1, color="rgba(50,50,50,0.4)")
        main_trace.text = texts
        main_trace.mode = "markers+text"
        main_trace.textposition = "middle center"
        main_trace.hovertemplate = "%{text}<extra></extra>"

        # Layout tweaks so it fills the panel better
        fig.update_layout(
            autosize=True,
            margin=dict(l=10, r=10, t=50, b=10),
            showlegend=False,
        )

        return fig.to_html(full_html=False)

    except Exception as e:
        print("⚠️ Error generating topic map:", e)
        return "<p>No topic map available.</p>"


# ============================================
# 🚀 Main Dashboard Function
# ============================================
def generate_dashboard():
    print("🚀 Generating dashboard...")

    docs, topic_summaries, topic_model, embeddings, theme_scores = generate_topic_results()
    total_docs = len(docs)

    if not docs or topic_model is None:
        print("⚠️ No docs or topic model – writing fallback dashboard.")
        output_path = os.path.join(OUTPUT_DIR, "index.html")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("<h3>No sufficient data to generate dashboard.</h3>")
        return

    # Topic persistence
    _save_topic_embeddings(embeddings, topic_summaries)

    # -------------------------------
    # Hero topics (top 5 by volume)
    # -------------------------------
    sorted_topics = sorted(
        topic_summaries.items(),
        key=lambda kv: kv[1].get("article_count", 0),
        reverse=True,
    )
    hero_topic_ids = [tid for tid, _ in sorted_topics[:5]]

    # Topic map HTML
    html_topic_map = _build_topic_map_html(topic_model, topic_summaries, hero_topic_ids)

    # -------------------------------
    # Theme metrics / deltas
    # -------------------------------
    prev = _load_previous_theme_signals()
    theme_metrics = {}

    for theme, info in theme_scores.items():
        theme_metrics[theme] = {
            "volume": info.get("volume", 0),
            "centrality": info.get("centrality", 0.0),
            # carry through article ids for heatmap
            "articles_raw": info.get("articles_raw", []),
        }

    # % change topicality
    for theme, m in theme_metrics.items():
        prev_vol = prev.get(theme, {}).get("volume", 0)
        if prev_vol:
            m["topicality"] = (m["volume"] - prev_vol) / prev_vol
        else:
            m["topicality"] = 0.0

    # Rankings
    for metric in ["centrality", "topicality"]:
        ordered = sorted(theme_metrics.items(), key=lambda x: -x[1][metric])
        for r, (theme, _) in enumerate(ordered, start=1):
            theme_metrics[theme][f"{metric}_rank"] = r

    # Build theme_signals for template (rounded + prior values)
    theme_signals = {
        t: {
            "centrality": round(m["centrality"], 3),
            "topicality": round(m["topicality"], 3),
            "centrality_rank": m["centrality_rank"],
            "topicality_rank": m["topicality_rank"],
            "prev_centrality": (
                round(prev.get(t, {}).get("centrality"), 3)
                if prev.get(t, {}).get("centrality") is not None
                else None
            ),
            "prev_topicality": prev.get(t, {}).get("topicality"),
            "prev_centrality_rank": prev.get(t, {}).get("centrality_rank"),
            "prev_topicality_rank": prev.get(t, {}).get("topicality_rank"),
        }
        for t, m in theme_metrics.items()
    }

    # Save today’s theme metrics (without article IDs)
    _save_theme_signals(theme_metrics)

    # Theme scatter HTML
    html_theme_map = _build_theme_map_html(theme_signals, total_docs)

    # Heatmap HTML (using full theme_metrics with article IDs)
    heatmap_html = _build_heatmap(topic_summaries, theme_metrics, topic_model)

    # Topic summaries for display
    summary_list = [
        {
            "topic_id": k,
            "title": v.get("title", ""),
            "summary": v.get("summary", "").replace("\n", "<br>"),
            "article_count": v.get("article_count", None),
        }
        for k, v in sorted(topic_summaries.items())
    ]

    # Render HTML
    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("dashboard_template.html")

    html = template.render(
        topic_map=html_topic_map,
        theme_map=html_theme_map,
        heatmap=heatmap_html,
        theme_signals=theme_signals,
        summaries=summary_list,
        hero_topic_ids=hero_topic_ids,
        run_date=os.getenv("RUN_DATE", "Today"),
        total_docs=total_docs,
    )

    with open(os.path.join(OUTPUT_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)

    print("🎉 Dashboard updated.")


if __name__ == "__main__":
    generate_dashboard()

