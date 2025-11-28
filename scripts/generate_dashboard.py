# ============================================
# 📄 generate_dashboard.py
# Version 5.6 – Stable with theme analytics
# ============================================

import os
import json
from jinja2 import Environment, FileSystemLoader

import plotly.graph_objects as go

from LDA_engine_with_BERTopic_v054 import generate_topic_results

# --------------------------------------------
# 1️⃣ OpenAI Key Validation
# --------------------------------------------
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("⚠️ OPENAI_API_KEY not found. Add it as a GitHub Secret.")

# --------------------------------------------
# 2️⃣ Output Directory & paths
# --------------------------------------------
OUTPUT_DIR = "dashboard"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOPIC_PERSISTENCE_JSON = os.path.join(OUTPUT_DIR, "yesterday_topics.json")
THEME_SIGNALS_JSON = os.path.join(OUTPUT_DIR, "yesterday_theme_signals.json")


def _save_topic_embeddings(embeddings, topic_summaries):
    """Save topic embeddings for persistence comparison tomorrow."""
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
        print(f"📁 Saved topic persistence file → {TOPIC_PERSISTENCE_JSON}")
    except Exception as e:
        print(f"❌ ERROR saving topic JSON file: {e}")


def _load_previous_theme_signals():
    if not os.path.exists(THEME_SIGNALS_JSON):
        print("🟡 No previous theme signals found – treating all themes as new.")
        return {}
    try:
        with open(THEME_SIGNALS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Error loading previous theme signals: {e}")
        return {}


def _save_theme_signals(theme_metrics):
    """Persist today's theme metrics for tomorrow's comparisons."""
    try:
        to_save = {}
        for theme, m in theme_metrics.items():
            to_save[theme] = {
                "volume": m.get("volume", 0),
                "centrality": m.get("centrality", 0.0),
                "topicality": m.get("topicality", 0.0),
                "centrality_rank": m.get("centrality_rank"),
                "topicality_rank": m.get("topicality_rank"),
            }
        with open(THEME_SIGNALS_JSON, "w", encoding="utf-8") as f:
            json.dump(to_save, f, indent=2)
        print(f"📁 Saved theme signals file → {THEME_SIGNALS_JSON}")
    except Exception as e:
        print(f"❌ ERROR saving theme signals JSON: {e}")


def _build_theme_map_html(theme_signals):
    """Create a simple theme map (centrality vs topicality) using Plotly."""
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
            title="Theme Distance Map (Centrality vs Δ Volume)",
            xaxis_title="Topicality (Δ news volume vs prior day)",
            yaxis_title="Centrality (overlapping themes)",
            margin=dict(l=40, r=40, t=40, b=40),
            height=400,
            width=600,
        )
        return fig.to_html(full_html=False)
    except Exception as e:
        print(f"⚠️ Error generating theme map: {e}")
        return "<p>No theme visualization available.</p>"


def generate_dashboard():
    print("🚀 Generating dashboard...")

    # Expect 5 outputs (last one now theme_scores)
    docs, topic_summaries, topic_model, embeddings, theme_scores = generate_topic_results()

    if not docs or not topic_model:
        print("⚠️ Insufficient data for full dashboard. Using fallback layout.")
        fallback_path = os.path.join(OUTPUT_DIR, "index.html")
        with open(fallback_path, "w", encoding="utf-8") as f:
            f.write("<h3>No sufficient data to generate dashboard today.</h3>")
        print(f"🟡 Dashboard fallback written → {fallback_path}")
        return

    # --------------------------------------------
    # 🔹 Save embeddings for tomorrow comparison
    # --------------------------------------------
    _save_topic_embeddings(embeddings, topic_summaries)

    # --------------------------------------------
    # 3️⃣ Build BERTopic Topic Map (unchanged)
    # --------------------------------------------
    try:
        fig_topics = topic_model.visualize_topics(width=600, height=650)
        html_topic_map = fig_topics.to_html(full_html=False)
    except Exception as e:
        print(f"⚠️ Error generating topic map: {e}")
        html_topic_map = "<p>No topic map available.</p>"

    # --------------------------------------------
    # 4️⃣ Theme analytics: centrality + topicality
    # --------------------------------------------
    theme_signals = {}
    if isinstance(theme_scores, dict) and theme_scores:
        # Base metrics for today
        theme_metrics = {}
        for theme, info in theme_scores.items():
            volume = info.get("volume", 0) or 0
            centrality = info.get("centrality", 0.0) or 0.0
            theme_metrics[theme] = {
                "volume": int(volume),
                "centrality": float(centrality),
            }

        # Load yesterday's theme metrics
        prev_theme_data = _load_previous_theme_signals()

        # Compute topicality (Δ volume vs prior day) and attach previous metrics
        for theme, m in theme_metrics.items():
            prev = prev_theme_data.get(theme, {})
            prev_volume = prev.get("volume", 0)
            prev_centrality = prev.get("centrality", 0.0)
            prev_topicality = prev.get("topicality", 0.0)

            topicality = m["volume"] - prev_volume

            m["topicality"] = float(topicality)
            m["prev_volume"] = prev_volume
            m["prev_centrality"] = float(prev_centrality)
            m["prev_topicality"] = float(prev_topicality)
            m["prev_centrality_rank"] = prev.get("centrality_rank")
            m["prev_topicality_rank"] = prev.get("topicality_rank")

        # Rankings for today
        centrality_sorted = sorted(
            theme_metrics.items(),
            key=lambda kv: (-kv[1]["centrality"], kv[0]),
        )
        topicality_sorted = sorted(
            theme_metrics.items(),
            key=lambda kv: (-kv[1]["topicality"], kv[0]),
        )

        for rank, (theme, _) in enumerate(centrality_sorted, start=1):
            theme_metrics[theme]["centrality_rank"] = rank
        for rank, (theme, _) in enumerate(topicality_sorted, start=1):
            theme_metrics[theme]["topicality_rank"] = rank

        # Build clean structure for template
        for theme, m in theme_metrics.items():
            theme_signals[theme] = {
                "centrality": round(m["centrality"], 2),
                "topicality": round(m["topicality"], 2),
                "centrality_rank": m.get("centrality_rank"),
                "topicality_rank": m.get("topicality_rank"),
                "prev_centrality": round(m["prev_centrality"], 2)
                if m.get("prev_centrality") is not None
                else None,
                "prev_topicality": round(m["prev_topicality"], 2)
                if m.get("prev_topicality") is not None
                else None,
                "prev_centrality_rank": m.get("prev_centrality_rank"),
                "prev_topicality_rank": m.get("prev_topicality_rank"),
            }

        # Save today's theme metrics for tomorrow
        _save_theme_signals(theme_metrics)

    # Theme Distance Map HTML
    html_theme_map = _build_theme_map_html(theme_signals)

    # --------------------------------------------
    # 5️⃣ Formatting data for template
    # --------------------------------------------
    summary_list = []
    for k, v in topic_summaries.items():
        if isinstance(v, dict):
            summary_list.append(
                {
                    "topic_id": k,
                    "title": v.get("title", ""),
                    "summary": v.get("summary", "").replace("\n", "<br>"),
                    "is_new": v.get("status", "").upper() == "NEW",
                    "is_persistent": v.get("status", "").upper() == "PERSISTENT",
                }
            )

    # --------------------------------------------
    # 6️⃣ Render via Jinja2
    # --------------------------------------------
    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("dashboard_template.html")

    rendered_html = template.render(
        topic_map=html_topic_map,
        theme_map=html_theme_map,
        theme_signals=theme_signals,
        summaries=summary_list,
        run_date=os.getenv("RUN_DATE", "Today"),
    )

    # --------------------------------------------
    # 7️⃣ Save HTML
    # --------------------------------------------
    output_path = os.path.join(OUTPUT_DIR, "index.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(rendered_html)

    print(f"🎉 Dashboard successfully written → {output_path}")


# --------------------------------------------
# Run manually
# --------------------------------------------
if __name__ == "__main__":
    generate_dashboard()
