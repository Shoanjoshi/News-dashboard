# ============================================
# 📄 generate_dashboard.py
# Version 5.8 – Updated metrics, centrality, and article counts
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
# 2️⃣ Output Directory
# --------------------------------------------
OUTPUT_DIR = "dashboard"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOPIC_PERSISTENCE_JSON = os.path.join(OUTPUT_DIR, "yesterday_topics.json")
THEME_SIGNALS_JSON = os.path.join(OUTPUT_DIR, "yesterday_theme_signals.json")


# --------------------------------------------
# Helper functions
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
    """Persist today's theme metrics for tomorrow."""
    try:
        with open(THEME_SIGNALS_JSON, "w", encoding="utf-8") as f:
            json.dump(theme_metrics, f, indent=2)
        print(f"📁 Saved theme metrics → {THEME_SIGNALS_JSON}")
    except Exception as e:
        print(f"❌ Error saving theme metrics: {e}")


def _stretch_figure(fig):
    """Make Plotly figures fill container (HTML controls size)."""
    fig.update_layout(
        autosize=True,
        height=None,
        width=None,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


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
            title=f"Theme Distance Map (Centrality vs Δ Volume) – {total_docs} articles",
            xaxis_title="Topicality (Δ news volume vs prior day)",
            yaxis_title="Centrality",
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

    # Expect: docs, topic_summaries, topic_model, embeddings, theme_scores
    docs, topic_summaries, topic_model, embeddings, theme_scores = generate_topic_results()
    total_docs = len(docs)

    if not docs or not topic_model:
        output_path = os.path.join(OUTPUT_DIR, "index.html")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("<h3>No sufficient data to generate dashboard.</h3>")
        print("⚠️ Dashboard fallback created.")
        return

    # 1️⃣ Save topic embeddings for persistence logic
    _save_topic_embeddings(embeddings, topic_summaries)

    # 2️⃣ Topic map
    try:
        fig_topics = topic_model.visualize_topics()
        fig_topics = _stretch_figure(fig_topics)
        html_topic_map = fig_topics.to_html(full_html=False)
    except Exception as e:
        print(f"⚠️ Error generating topic map: {e}")
        html_topic_map = "<p>No topic map available.</p>"

    # 3️⃣ Theme metrics – start from engine output
    prev_data = _load_previous_theme_signals()
    theme_metrics = {}

    # Base from engine's theme_scores
    for theme, info in theme_scores.items():
        theme_metrics[theme] = {
            "volume": int(info.get("volume", 0)),
            "centrality": float(info.get("centrality", 0.0)),
        }

    # Ensure "Others" theme exists & totals add up
    total_theme_volume = sum(v["volume"] for v in theme_metrics.values())
    if total_docs > total_theme_volume:
        # Add or adjust Others
        others_volume = total_docs - total_theme_volume
        if "Others" in theme_metrics:
            theme_metrics["Others"]["volume"] += others_volume
        else:
            theme_metrics["Others"] = {
                "volume": others_volume,
                "centrality": 0.0,
            }

    # 4️⃣ Topicality = Δ volume vs prior day (keep as requested)
    for theme, m in theme_metrics.items():
        prev_volume = prev_data.get(theme, {}).get("volume", 0)
        m["topicality"] = m["volume"] - prev_volume

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
            "prev_centrality": prev_data.get(theme, {}).get("centrality"),
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

    # 9️⃣ Prepare summaries for template
    summary_list = [
        {
            "topic_id": k,
            "title": v.get("title", ""),
            "summary": v.get("summary", "").replace("\n", "<br>"),
            "article_count": v.get("article_count", None),   # NEW FIELD
            "is_new": v.get("status") == "NEW",
            "is_persistent": v.get("status") == "PERSISTENT",
        }
        for k, v in topic_summaries.items()
    ]


    # 🔟 Render dashboard
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
