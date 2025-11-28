# ============================================
# 📄 generate_dashboard.py
# Version 5.7 – Optimized for large charts in 4×4 layout
# ============================================

import os
import json
from jinja2 import Environment, FileSystemLoader
import plotly.graph_objects as go
from LDA_engine_with_BERTopic_v054.py import generate_topic_results

# --------------------------------------------
# 🔐 OpenAI Key Validation
# --------------------------------------------
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("⚠️ OPENAI_API_KEY not found. Add it as a GitHub Secret.")

# --------------------------------------------
# 📂 Output Directory & File Paths
# --------------------------------------------
OUTPUT_DIR = "dashboard"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOPIC_PERSISTENCE_JSON = os.path.join(OUTPUT_DIR, "yesterday_topics.json")
THEME_SIGNALS_JSON = os.path.join(OUTPUT_DIR, "yesterday_theme_signals.json")


def _save_topic_embeddings(embeddings, topic_summaries):
    """Store embeddings for tomorrow’s persistence comparison."""
    try:
        data_to_save = {
            str(k): {
                "embedding": embeddings[k],
                "title": topic_summaries[k].get("title", ""),
            }
            for k in embeddings.keys()
        }
        with open(TOPIC_PERSISTENCE_JSON, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, indent=2)
        print(f"📁 Saved topic persistence → {TOPIC_PERSISTENCE_JSON}")
    except Exception as e:
        print(f"❌ Error saving topic JSON: {e}")


def _load_previous_theme_signals():
    """Load yesterday’s theme metrics if available."""
    if not os.path.exists(THEME_SIGNALS_JSON):
        print("🟡 No previous theme signals found – treating as first run.")
        return {}
    try:
        with open(THEME_SIGNALS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Error loading previous theme data: {e}")
        return {}


def _save_theme_signals(theme_metrics):
    """Persist today’s theme metrics for next run."""
    try:
        to_save = {
            theme: {
                "volume": m.get("volume", 0),
                "centrality": m.get("centrality", 0.0),
                "topicality": m.get("topicality", 0.0),
                "centrality_rank": m.get("centrality_rank"),
                "topicality_rank": m.get("topicality_rank"),
            }
            for theme, m in theme_metrics.items()
        }
        with open(THEME_SIGNALS_JSON, "w", encoding="utf-8") as f:
            json.dump(to_save, f, indent=2)
        print(f"📁 Saved theme metrics → {THEME_SIGNALS_JSON}")
    except Exception as e:
        print(f"❌ Error saving theme JSON: {e}")


def _build_theme_map_html(theme_signals):
    """Generate responsive Theme Distance Map."""
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
                    textposition="top center"
                )
            ]
        )

        fig.update_layout(
            title="Theme Distance Map (Centrality vs Δ Volume)",
            xaxis_title="Topicality (Δ volume vs prior day)",
            yaxis_title="Centrality (theme overlap count)",
            autosize=True,
            height=None,
            width=None,
            margin=dict(l=10, r=10, t=50, b=10),
        )

        return fig.to_html(full_html=False)
    except Exception as e:
        print(f"⚠️ Error generating theme map: {e}")
        return "<p>No theme visualization available.</p>"


def _stretch_figure_layout(fig):
    """Ensure Plotly figures fill their containers."""
    fig.update_layout(
        autosize=True,
        height=None,  # Allow CSS to determine size
        width=None,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def generate_dashboard():
    print("🚀 Generating dashboard...")

    docs, topic_summaries, topic_model, embeddings, theme_scores = generate_topic_results()

    if not docs or not topic_model:
        fallback_path = os.path.join(OUTPUT_DIR, "index.html")
        with open(fallback_path, "w", encoding="utf-8") as f:
            f.write("<h3>No sufficient data to generate dashboard today.</h3>")
        print(f"🟡 Dashboard fallback → {fallback_path}")
        return

    # 1️⃣ Save topic embeddings
    _save_topic_embeddings(embeddings, topic_summaries)

    # 2️⃣ Topic Map (expanded)
    try:
        fig_topics = topic_model.visualize_topics()  # removed fixed size
        fig_topics = _stretch_figure_layout(fig_topics)
        html_topic_map = fig_topics.to_html(full_html=False)
    except Exception as e:
        print(f"⚠️ Error generating topic map: {e}")
        html_topic_map = "<p>No topic map available.</p>"

    # 3️⃣ Theme analytics logic (unchanged logic)
    theme_signals = {}
    if isinstance(theme_scores, dict) and theme_scores:
        theme_metrics = {
            theme: {
                "volume": int(info.get("volume", 0)),
                "centrality": float(info.get("centrality", 0.0)),
            }
            for theme, info in theme_scores.items()
        }

        prev = _load_previous_theme_signals()

        for theme, m in theme_metrics.items():
            prev_volume = prev.get(theme, {}).get("volume", 0)
            prev_centrality = prev.get(theme, {}).get("centrality", 0)
            prev_topicality = prev.get(theme, {}).get("topicality", 0)

            m["topicality"] = m["volume"] - prev_volume
            m["prev_volume"] = prev_volume
            m["prev_centrality"] = prev_centrality
            m["prev_topicality"] = prev_topicality
            m["prev_centrality_rank"] = prev.get(theme, {}).get("centrality_rank")
            m["prev_topicality_rank"] = prev.get(theme, {}).get("topicality_rank")

        sorted_cent = sorted(theme_metrics.items(), key=lambda kv: -kv[1]["centrality"])
        sorted_topic = sorted(theme_metrics.items(), key=lambda kv: -kv[1]["topicality"])

        for i, (theme, _) in enumerate(sorted_cent, 1):
            theme_metrics[theme]["centrality_rank"] = i
        for i, (theme, _) in enumerate(sorted_topic, 1):
            theme_metrics[theme]["topicality_rank"] = i

        theme_signals = {
            theme: {
                "centrality": round(m["centrality"], 2),
                "topicality": round(m["topicality"], 2),
                "centrality_rank": m["centrality_rank"],
                "topicality_rank": m["topicality_rank"],
                "prev_centrality": round(m["prev_centrality"], 2),
                "prev_topicality": round(m["prev_topicality"], 2),
                "prev_centrality_rank": m.get("prev_centrality_rank"),
                "prev_topicality_rank": m.get("prev_topicality_rank"),
            }
            for theme, m in theme_metrics.items()
        }

        _save_theme_signals(theme_metrics)

    # 4️⃣ Theme visualization (expanded)
    html_theme_map = _build_theme_map_html(theme_signals)

    # 5️⃣ Format summaries
    summary_list = [
        {
            "topic_id": k,
            "title": v.get("title", ""),
            "summary": v.get("summary", "").replace("\n", "<br>"),
            "is_new": v.get("status", "").upper() == "NEW",
            "is_persistent": v.get("status", "").upper() == "PERSISTENT",
        }
        for k, v in topic_summaries.items()
    ]

    # 6️⃣ Render HTML
    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("dashboard_template.html")

    rendered_html = template.render(
        topic_map=html_topic_map,
        theme_map=html_theme_map,
        theme_signals=theme_signals,
        summaries=summary_list,
        run_date=os.getenv("RUN_DATE", "Today"),
    )

    # 7️⃣ Save output
    output_path = os.path.join(OUTPUT_DIR, "index.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(rendered_html)

    print(f"🎉 Dashboard saved → {output_path}")


if __name__ == "__main__":
    generate_dashboard()
