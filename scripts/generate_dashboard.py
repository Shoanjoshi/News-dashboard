# ============================================
# 📄 generate_dashboard.py
# Version 6.x – Hero topics + cleaned styling
# ============================================

import os
import json

import numpy as np
from jinja2 import Environment, FileSystemLoader
import plotly.graph_objects as go

from LDA_engine_with_BERTopic_v054 import generate_topic_results, THEMES

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


# --------------------------------------------
# Helper: Save topic embeddings
# --------------------------------------------
def _save_topic_embeddings(embeddings, topic_summaries):
    """Persist topic embeddings for NEW/PERSISTENT status tomorrow."""
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
        print(f"📁 Saved topic persistence file → {TOPIC_PERSISTENCE_JSON}")
    except Exception as e:
        print(f"❌ Error saving topic JSON: {e}")


# --------------------------------------------
# Helper: Load prior theme metrics
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


# --------------------------------------------
# Helper: Save today's theme metrics
# --------------------------------------------
def _save_theme_signals(theme_metrics):
    """Persist today's theme metrics for tomorrow."""
    try:
        with open(THEME_SIGNALS_JSON, "w", encoding="utf-8") as f:
            json.dump(theme_metrics, f, indent=2)
        print(f"📁 Saved theme metrics → {THEME_SIGNALS_JSON}")
    except Exception as e:
        print(f"❌ Error saving theme metrics: {e}")


# --------------------------------------------
# Helper: Make Plotly figures stretch
# --------------------------------------------
def _stretch_figure(fig):
    """Make Plotly figures fill container (HTML controls size)."""
    fig.update_layout(
        autosize=True,
        height=650,
        width=None,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


# --------------------------------------------
# Helper: Customize Topic Map with hero topics
# --------------------------------------------
def _customize_topic_map(fig, topic_summaries, hero_topic_ids):
    """
    - All topics: label as T<ID>
    - Hero topics (top 5 by volume): larger, light-brown bubbles + 'T<ID> – title'
    """
    try:
        if not fig.data:
            return fig

        scatter = fig.data[0]
        n_points = len(scatter.x)
        if n_points == 0:
            return fig

        # --- Marker sizes --------------------------------------------------
        base_size = scatter.marker.size
        if base_size is None:
            sizes = np.full(n_points, 40.0)
        else:
            arr = np.array(base_size, dtype=float)
            if arr.ndim == 0:
                sizes = np.full(n_points, float(arr))
            elif arr.size == 1:
                sizes = np.full(n_points, float(arr[0]))
            else:
                sizes = arr

        # --- Colors & text -------------------------------------------------
        base_color = "rgba(140, 160, 175, 0.55)"   # soft grey-blue
        hero_color = "rgba(201, 155, 90, 0.9)"     # light brown
        colors = [base_color] * n_points

        if hasattr(scatter, "text") and scatter.text is not None:
            texts = list(scatter.text)
        else:
            texts = [""] * n_points

        # --- Map each point to a topic_id ---------------------------------
        customdata = getattr(scatter, "customdata", None)
        topic_ids_for_point = [None] * n_points

        if customdata is not None:
            for idx, row in enumerate(customdata):
                topic_id = None
                if isinstance(row, (list, tuple, np.ndarray)):
                    # search from the end for something that casts to int
                    for val in row[::-1]:
                        try:
                            topic_id = int(val)
                            break
                        except (TypeError, ValueError):
                            continue
                else:
                    try:
                        topic_id = int(row)
                    except (TypeError, ValueError):
                        pass
                topic_ids_for_point[idx] = topic_id

        # Fallback: simple mapping by sorted topic ids
        if all(tid is None for tid in topic_ids_for_point):
            ordered_topic_ids = sorted(topic_summaries.keys())
            for idx, tid in enumerate(ordered_topic_ids[:n_points]):
                topic_ids_for_point[idx] = tid

        # --- Short titles for hero topics ---------------------------------
        short_titles = {}
        for tid in hero_topic_ids:
            title = topic_summaries.get(tid, {}).get("title", "").strip()
            if title:
                short_titles[tid] = (title[:45] + "…") if len(title) > 45 else title
            else:
                short_titles[tid] = ""

        # --- Apply styling -------------------------------------------------
        for idx, tid in enumerate(topic_ids_for_point):
            if tid is None:
                continue

            base_label = f"T{tid}"

            if tid in hero_topic_ids:
                colors[idx] = hero_color
                sizes[idx] = sizes[idx] * 1.25
                title = short_titles.get(tid, "")
                texts[idx] = f"{base_label} – {title}" if title else base_label
            else:
                texts[idx] = base_label

        scatter.marker.size = sizes.tolist()
        scatter.marker.color = colors
        scatter.text = texts
        scatter.textposition = "top center"

    except Exception as e:
        # Styling should never break the run
        print(f"⚠️ Error customizing topic map: {e}")

    return fig


# --------------------------------------------
# Build Theme Scatter Plot HTML
# --------------------------------------------
def _build_theme_map_html(theme_signals, total_docs):
    """Build Theme Distance Map HTML."""
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
            xaxis_title="Topicality (% change)",
            yaxis_title="Centrality",
            autosize=True,
        )

        return fig.to_html(full_html=False)

    except Exception as e:
        print(f"⚠️ Error generating theme map: {e}")
        return "<p>No theme visualization available.</p>"


# --------------------------------------------
# Build Topic × Theme Heatmap
# --------------------------------------------
def _build_heatmap(topic_summaries, theme_metrics, docs, topic_model):
    """Creates heatmap: rows=topic title, cols=themes, values=article counts."""
    try:
        ordered_topics = sorted(topic_summaries.keys())
        row_labels = [topic_summaries[t]["title"] for t in ordered_topics]
        col_labels = THEMES + ["Others"]

        heat = np.zeros((len(ordered_topics), len(col_labels)), dtype=int)

        # Recompute article → topic mapping
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
                colorbar=dict(lenmode="fraction", len=0.8),
            )
        )

        fig.update_layout(
            title="Topic × Theme Volume Heatmap",
            xaxis_nticks=len(col_labels),
            yaxis_autorange="reversed",
            autosize=True,
        )

        return fig.to_html(full_html=False)

    except Exception as e:
        print(f"⚠️ Heatmap error: {e}")
        return "<p>Heatmap unavailable.</p>"


# ============================================
# 🚀 Main Dashboard Function
# ============================================
def generate_dashboard():
    print("🚀 Generating dashboard...")

    # Expect: docs, topic_summaries, topic_model, embeddings, theme_scores
    docs, topic_summaries, topic_model, embeddings, theme_scores = generate_topic_results()
    total_docs = len(docs)

    if not docs or not topic_model:
        output_path = os.path.join(OUTPUT_DIR, "index.html")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("<h3>No sufficient data to generate dashboard.</h3>")
        print("⚠️ Dashboard fallback created.")
        return

    # ⭐ Determine hero topics (top 5 by article count)
    topic_volumes = [
        (tid, topic_summaries[tid].get("article_count", 0))
        for tid in topic_summaries
    ]
    topic_volumes.sort(key=lambda x: -x[1])
    hero_topic_ids = {tid for tid, _ in topic_volumes[:5]}
    print(f"⭐ Hero topics (by volume): {sorted(hero_topic_ids)}")

    # 1️⃣ Save topic embeddings for persistence logic
    _save_topic_embeddings(embeddings, topic_summaries)

    # 2️⃣ Topic map – customize with hero topics
    try:
        fig_topics = topic_model.visualize_topics(width=900, height=650)
        fig_topics = _stretch_figure(fig_topics)
        fig_topics = _customize_topic_map(fig_topics, topic_summaries, hero_topic_ids)
        html_topic_map = fig_topics.to_html(full_html=False)
    except Exception as e:
        print(f"⚠️ Error generating topic map: {e}")
        html_topic_map = "<p>No topic map available.</p>"

    # 3️⃣ Theme metrics – start from engine output
    prev_data = _load_previous_theme_signals()
    theme_metrics = {}

    # Base from engine's theme_scores (volume + centrality)
    for theme, info in theme_scores.items():
        theme_metrics[theme] = {
            "volume": int(info.get("volume", 0)),
            "centrality": float(info.get("centrality", 0.0)),
        }

    # Ensure "Others" theme exists & totals add up
    total_theme_volume = sum(v["volume"] for v in theme_metrics.values())
    if total_docs > total_theme_volume:
        others_volume = total_docs - total_theme_volume
        if "Others" in theme_metrics:
            theme_metrics["Others"]["volume"] += others_volume
        else:
            theme_metrics["Others"] = {
                "volume": others_volume,
                "centrality": 0.0,
            }

    # 4️⃣ Topicality = % change in volume vs prior day
    for theme, m in theme_metrics.items():
        prev_volume = prev_data.get(theme, {}).get("volume", 0)
        if prev_volume > 0:
            m["topicality"] = (m["volume"] - prev_volume) / prev_volume
        else:
            m["topicality"] = 0.0

    # 5️⃣ Rank ordering
    for metric in ["centrality", "topicality"]:
        ranked = sorted(theme_metrics.items(), key=lambda x: -x[1][metric])
        for rank, (theme, _) in enumerate(ranked, start=1):
            theme_metrics[theme][f"{metric}_rank"] = rank

    # 6️⃣ Build theme_signals for template, with gentle rounding
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

    # 7️⃣ Save today's theme metrics for tomorrow
    _save_theme_signals(theme_metrics)

    # 8️⃣ Theme map HTML
    html_theme_map = _build_theme_map_html(theme_signals, total_docs)

    # 9️⃣ Heatmap HTML
    heatmap_html = _build_heatmap(topic_summaries, theme_scores, docs, topic_model)

    # 🔟 Prepare summaries for template (no NEW/PERSISTENT coloring logic)
    summary_list = [
        {
            "topic_id": k,
            "title": v.get("title", ""),
            "summary": v.get("summary", "").replace("\n", "<br>"),
            "article_count": v.get("article_count", None),
            "is_new": v.get("status") == "NEW",
            "is_persistent": v.get("status") == "PERSISTENT",
        }
        for k, v in topic_summaries.items()
    ]

    # Render dashboard
    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("dashboard_template.html")
    rendered_html = template.render(
        topic_map=html_topic_map,
        theme_map=html_theme_map,
        heatmap=heatmap_html,
        theme_signals=theme_signals,
        summaries=summary_list,
        hero_topic_ids=sorted(hero_topic_ids),
        run_date=os.getenv("RUN_DATE", "Today"),
        total_docs=total_docs,
    )

    output_path = os.path.join(OUTPUT_DIR, "index.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(rendered_html)
    print("🎉 Dashboard updated successfully!")


if __name__ == "__main__":
    generate_dashboard()
