# ============================================
# 📄 generate_dashboard.py
# Custom topic map + theme signals + heatmap
# ============================================

import os
import json
import math

from jinja2 import Environment, FileSystemLoader
import plotly.graph_objects as go
import numpy as np

from LDA_engine_with_BERTopic_v054 import generate_topic_results, THEMES

# --------------------------------------------
# 1️⃣ OpenAI Key Validation
# --------------------------------------------
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("⚠️ OPENAI_API_KEY not found. Add it as a GitHub Secret.")

# --------------------------------------------
# 2️⃣ Output Directory / Persistence Files
# --------------------------------------------
OUTPUT_DIR = "dashboard"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOPIC_PERSISTENCE_JSON = os.path.join(OUTPUT_DIR, "yesterday_topics.json")
THEME_SIGNALS_JSON = os.path.join(OUTPUT_DIR, "yesterday_theme_signals.json")


# --------------------------------------------
# Helper: Save topic embeddings for persistence
# --------------------------------------------
def _save_topic_embeddings(embeddings, topic_summaries):
    """Persist topic embeddings for future NEW/PERSISTENT logic."""
    try:
        payload = {
            str(k): {
                "embedding": embeddings[k],
                "title": topic_summaries[k].get("title", ""),
            }
            for k in embeddings
        }
        with open(TOPIC_PERSISTENCE_JSON, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"📁 Saved topic persistence file → {TOPIC_PERSISTENCE_JSON}")
    except Exception as e:
        print(f"❌ Error saving topic JSON: {e}")


# --------------------------------------------
# Helper: Theme metrics persistence
# --------------------------------------------
def _load_previous_theme_signals():
    """Load yesterday's theme metrics for topicality deltas."""
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
    """Persist ONLY volume + centrality for tomorrow's calculations."""
    try:
        to_save = {
            theme: {
                "volume": int(m.get("volume", 0)),
                "centrality": float(m.get("centrality", 0.0)),
            }
            for theme, m in theme_metrics.items()
        }
        with open(THEME_SIGNALS_JSON, "w", encoding="utf-8") as f:
            json.dump(to_save, f, indent=2)
        print(f"📁 Saved theme metrics → {THEME_SIGNALS_JSON}")
    except Exception as e:
        print(f"❌ Error saving theme metrics: {e}")


# --------------------------------------------
# Helper: Stretch figures to panel
# --------------------------------------------
def _stretch_figure(fig):
    fig.update_layout(
        autosize=True,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


# --------------------------------------------
# 3️⃣ Topic Map – custom Plotly scatter
# --------------------------------------------
def _build_topic_map_html(topic_summaries, embeddings, total_docs):
    """
    Build a custom topic map:
      • bubbles sized by article_count
      • top 5 topics highlighted + labeled with TITLE
      • all others labeled only as T{id}
    """
    try:
        if not embeddings or not topic_summaries:
            return "<p>No topic map available.</p>"

        # Sort topics by ID for stable ordering
        topic_ids = sorted(topic_summaries.keys())

        # Compute volumes to find top-5 topics
        volumes = [
            (tid, topic_summaries[tid].get("article_count", 0)) for tid in topic_ids
        ]
        volumes_sorted = sorted(volumes, key=lambda x: -x[1])
        top_n = 5
        hero_ids = {tid for tid, _ in volumes_sorted[:top_n]}

        xs, ys, sizes, colors, labels, hover_texts = [], [], [], [], [], []

        # Color palette
        hero_color = "rgba(194, 136, 64, 0.85)"  # light brown / amber
        base_color = "rgba(140, 160, 180, 0.65)"  # muted blue-grey

        for tid in topic_ids:
            emb = embeddings.get(tid)
            if not emb or len(emb) < 2:
                continue

            x, y = emb[0], emb[1]
            topic_info = topic_summaries[tid]
            count = topic_info.get("article_count", 0)
            title = topic_info.get("title", f"TOPIC {tid}")

            # Bubble size: sublinear in article count to avoid extremes
            size = 18 + 3.5 * math.sqrt(max(count, 1))

            # Label logic
            if tid in hero_ids:
                label = f"T{tid}: {title}"
                color = hero_color
            else:
                label = f"T{tid}"
                color = base_color

            hover = f"T{tid} · {title}<br>Articles: {count}"

            xs.append(x)
            ys.append(y)
            sizes.append(size)
            colors.append(color)
            labels.append(label)
            hover_texts.append(hover)

        if not xs:
            return "<p>No topic map available.</p>"

        fig = go.Figure(
            data=[
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="markers+text",
                    text=labels,
                    textposition="top center",
                    textfont=dict(size=11),
                    marker=dict(
                        size=sizes,
                        color=colors,
                        line=dict(color="rgba(40,40,40,0.5)", width=1),
                    ),
                    hovertext=hover_texts,
                    hoverinfo="text",
                )
            ]
        )

        fig.update_layout(
            title=f"Intertopic Distance Map – {total_docs} articles",
            xaxis_title=None,
            yaxis_title=None,
            xaxis=dict(showticklabels=False, zeroline=False),
            yaxis=dict(showticklabels=False, zeroline=False),
        )

        fig = _stretch_figure(fig)
        return fig.to_html(full_html=False)

    except Exception as e:
        print(f"⚠️ Error generating custom topic map: {e}")
        return "<p>No topic map available.</p>"


# --------------------------------------------
# 4️⃣ Theme Distance Map (scatter)
# --------------------------------------------
def _build_theme_map_html(theme_signals, total_docs):
    if not theme_signals:
        return "<p>No theme visualization available.</p>"

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
            xaxis_title="Topicality (% change vs prior day)",
            yaxis_title="Centrality",
            autosize=True,
        )

        fig = _stretch_figure(fig)
        return fig.to_html(full_html=False)

    except Exception as e:
        print(f"⚠️ Error generating theme scatter: {e}")
        return "<p>No theme visualization available.</p>"


# --------------------------------------------
# 5️⃣ Topic × Theme Heatmap
# --------------------------------------------
def _build_heatmap(topic_summaries, theme_metrics, topic_model):
    """Heatmap of article counts by (topic, theme)."""
    try:
        if not topic_summaries or topic_model is None:
            return "<p>Heatmap unavailable.</p>"

        ordered_topics = sorted(topic_summaries.keys())
        topic_index = {tid: i for i, tid in enumerate(ordered_topics)}

        row_labels = [topic_summaries[t]["title"] for t in ordered_topics]
        col_labels = THEMES + ["Others"]

        # Prepare article sets per theme
        theme_articles = {
            theme: set(info.get("articles_raw", []))
            for theme, info in theme_metrics.items()
        }

        # Matrix [topic x theme]
        heat = np.zeros((len(ordered_topics), len(col_labels)), dtype=int)

        # BERTopic stores per-document topic assignment here
        doc_topics = getattr(topic_model, "topics_", None)
        if doc_topics is None:
            return "<p>Heatmap unavailable.</p>"

        for article_id, topic_id in enumerate(doc_topics):
            row = topic_index.get(topic_id)
            if row is None:
                continue
            for col, theme in enumerate(col_labels):
                if article_id in theme_articles.get(theme, set()):
                    heat[row, col] += 1

        fig = go.Figure(
            data=go.Heatmap(
                z=heat,
                x=col_labels,
                y=row_labels,
                colorscale="Blues",
                colorbar=dict(lenmode="fraction", len=0.8),
                showscale=True,
            )
        )

        fig.update_layout(
            title="Topic × Theme Volume Heatmap",
            xaxis_nticks=len(col_labels),
            yaxis_autorange="reversed",
        )

        fig = _stretch_figure(fig)
        return fig.to_html(full_html=False)

    except Exception as e:
        print(f"⚠️ Heatmap error: {e}")
        return "<p>Heatmap unavailable.</p>"


# ============================================
# 🚀 Main Dashboard Generator
# ============================================
def generate_dashboard():
    print("🚀 Generating dashboard...")

    # Engine returns: docs, topic_summaries, topic_model, embeddings, theme_metrics_raw
    docs, topic_summaries, topic_model, embeddings, theme_metrics_raw = generate_topic_results()
    total_docs = len(docs)

    if not docs or topic_model is None:
        output_path = os.path.join(OUTPUT_DIR, "index.html")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("<h3>No sufficient data to generate dashboard.</h3>")
        print("⚠️ Dashboard fallback created.")
        return

    # 1️⃣ Save topic embeddings
    _save_topic_embeddings(embeddings, topic_summaries)

    # 2️⃣ Prepare theme metrics (keep article lists for heatmap only)
    prev_data = _load_previous_theme_signals()
    theme_metrics = {}

    for theme, info in theme_metrics_raw.items():
        theme_metrics[theme] = {
            "volume": int(info.get("volume", 0)),
            "centrality": float(info.get("centrality", 0.0)),
            "articles_raw": info.get("articles_raw", []),
        }

    # Ensure "Others" theme exists
    if "Others" not in theme_metrics:
        theme_metrics["Others"] = {"volume": 0, "centrality": 0.0, "articles_raw": []}

    # 3️⃣ Topicality = % change in volume vs prior day
    for theme, m in theme_metrics.items():
        prev_volume = prev_data.get(theme, {}).get("volume", 0)
        if prev_volume > 0:
            m["topicality"] = (m["volume"] - prev_volume) / prev_volume
        else:
            m["topicality"] = 0.0

    # 4️⃣ Rank ordering
    for metric in ["centrality", "topicality"]:
        ranked = sorted(theme_metrics.items(), key=lambda x: -x[1][metric])
        for rank, (theme, _) in enumerate(ranked, start=1):
            theme_metrics[theme][f"{metric}_rank"] = rank

    # 5️⃣ Build theme_signals for template (rounded + prior metrics)
    theme_signals = {
        theme: {
            "centrality": round(m["centrality"], 3),
            "topicality": round(m["topicality"], 3),
            "centrality_rank": m.get("centrality_rank"),
            "topicality_rank": m.get("topicality_rank"),
            "volume": m.get("volume", 0),
            "prev_centrality": (
                round(prev_data.get(theme, {}).get("centrality"), 3)
                if prev_data.get(theme, {}).get("centrality") is not None
                else None
            ),
            "prev_topicality": prev_data.get(theme, {}).get("topicality"),
            "prev_centrality_rank": prev_data.get(theme, {}).get("centrality_rank"),
            "prev_topicality_rank": prev_data.get(theme, {}).get("topicality_rank"),
        }
        for theme, m in theme_metrics.items()
    }

    # 6️⃣ Persist today's theme metrics (volume + centrality only)
    _save_theme_signals(theme_metrics)

    # 7️⃣ Build visual components
    html_topic_map = _build_topic_map_html(topic_summaries, embeddings, total_docs)
    html_theme_map = _build_theme_map_html(theme_signals, total_docs)
    html_heatmap = _build_heatmap(topic_summaries, theme_metrics, topic_model)

    # 8️⃣ Topic summaries list for template
    #     (rows 0–4 will be shaded in template; no NEW/PERSISTENT flags now)
    summary_list = [
        {
            "topic_id": k,
            "title": v.get("title", ""),
            "summary": v.get("summary", "").replace("\n", "<br>"),
            "article_count": v.get("article_count", None),
        }
        for k, v in sorted(topic_summaries.items(), key=lambda kv: kv[0])
    ]

    # 9️⃣ Render dashboard via Jinja template
    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("dashboard_template.html")

    rendered_html = template.render(
        topic_map=html_topic_map,
        theme_map=html_theme_map,
        heatmap=html_heatmap,
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

