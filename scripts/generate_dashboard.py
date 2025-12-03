
# ============================================
# generate_dashboard.py — Clean Version with Topic Map Fix
# ============================================

import os
import json
from jinja2 import Environment, FileSystemLoader
import plotly.graph_objects as go
import numpy as np

from LDA_engine_with_BERTopic_v054 import generate_topic_results, THEMES

OUTPUT_DIR = "dashboard"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOPIC_PERSISTENCE_JSON = os.path.join(OUTPUT_DIR, "yesterday_topics.json")
THEME_SIGNALS_JSON = os.path.join(OUTPUT_DIR, "yesterday_theme_signals.json")


def _save_topic_embeddings(embeddings, summaries):
    with open(TOPIC_PERSISTENCE_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {str(k): {"embedding": embeddings[k], "title": summaries[k].get("title", "")}
             for k in embeddings},
            f,
            indent=2
        )


def _load_previous_theme_signals():
    if not os.path.exists(THEME_SIGNALS_JSON):
        return {}
    try:
        with open(THEME_SIGNALS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}


def _save_theme_signals(theme_metrics):
    cleaned = {
        t: {
            "volume": int(m["volume"]),
            "centrality": float(m["centrality"]),
            "topicality": float(m["topicality"]),
            "centrality_rank": m["centrality_rank"],
            "topicality_rank": m["topicality_rank"],
        }
        for t, m in theme_metrics.items()
    }
    with open(THEME_SIGNALS_JSON, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2)


def _build_theme_map_html(theme_signals, total_docs):
    themes = list(theme_signals.keys())
    xs = [theme_signals[t]["topicality"] for t in themes]
    ys = [theme_signals[t]["centrality"] for t in themes]

    fig = go.Figure(
        data=[go.Scatter(
            x=xs, y=ys, mode="markers+text", text=themes,
            textposition="top center"
        )]
    )

    fig.update_layout(
        title=f"Theme Distance Map – {total_docs} articles",
        xaxis_title="Topicality (% change)",
        yaxis_title="Centrality",
        autosize=True,
        margin=dict(l=40, r=20, t=60, b=40),
    )

    return fig.to_html(full_html=False)


def _build_heatmap(topic_summaries, theme_metrics, topics):
    ordered = sorted(topic_summaries.keys())
    rows = [topic_summaries[t]["title"] for t in ordered]
    cols = list(theme_metrics.keys())

    heat = np.zeros((len(ordered), len(cols)), dtype=int)

    for i, topic_id in enumerate(topics):
        if topic_id not in topic_summaries:
            continue
        row = ordered.index(topic_id)

        for j, theme in enumerate(cols):
            if i in theme_metrics[theme]["articles"]:
                heat[row, j] += 1

    fig = go.Figure(
        data=go.Heatmap(
            z=heat, x=cols, y=rows,
            colorscale="Blues", zmin=0, zmax=heat.max(),
            colorbar=dict(lenmode="fraction", len=0.8),
        )
    )

    fig.update_layout(
        title="Topic × Theme Volume Heatmap",
        margin=dict(l=80, r=20, t=60, b=40),
        xaxis_tickangle=-45,
    )

    return fig.to_html(full_html=False)


def generate_dashboard():
    docs, topic_summaries, topic_model, embeddings, theme_metrics = generate_topic_results()
    total_docs = len(docs)

    _save_topic_embeddings(embeddings, topic_summaries)
    prev = _load_previous_theme_signals()

    # compute % change topicality and ranks
    for theme, m in theme_metrics.items():
        prev_vol = prev.get(theme, {}).get("volume", 0)
        m["topicality"] = (m["volume"] - prev_vol) / prev_vol if prev_vol else 0.0

    for metric in ["centrality", "topicality"]:
        ordered = sorted(theme_metrics.items(), key=lambda x: -x[1][metric])
        for rank, (theme, _) in enumerate(ordered, start=1):
            theme_metrics[theme][f"{metric}_rank"] = rank

    theme_signals = {
        theme: {
            "centrality": round(m["centrality"], 3),
            "topicality": round(m["topicality"], 3),
            "centrality_rank": m["centrality_rank"],
            "topicality_rank": m["topicality_rank"],
            "prev_centrality": round(prev.get(theme, {}).get("centrality", 0), 3)
                              if prev.get(theme) else None,
            "prev_topicality": prev.get(theme, {}).get("topicality", None),
            "prev_centrality_rank": prev.get(theme, {}).get("centrality_rank", None),
            "prev_topicality_rank": prev.get(theme, {}).get("topicality_rank", None),
        }
        for theme, m in theme_metrics.items()
    }

    _save_theme_signals(theme_metrics)

    # Render plots
    topic_map_html = topic_model.visualize_topics().update_layout(
        margin=dict(l=20, r=20, t=60, b=20),
        autosize=True
    ).to_html(full_html=False)

    theme_map_html = _build_theme_map_html(theme_signals, total_docs)
    heatmap_html = _build_heatmap(topic_summaries, theme_metrics, topic_model.topics_)

    summary_list = [
        {
            "topic_id": k,
            "title": v["title"],
            "summary": v["summary"].replace("\n", "<br>"),
            "article_count": v["article_count"],
            "is_new": v.get("status") == "NEW",
            "is_persistent": v.get("status") == "PERSISTENT",
        }
        for k, v in topic_summaries.items()
    ]

    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("dashboard_template.html")
    html = template.render(
        topic_map=topic_map_html,
        theme_map=theme_map_html,
        heatmap=heatmap_html,
        theme_signals=theme_signals,
        summaries=summary_list,
        run_date=os.getenv("RUN_DATE", "Today"),
        total_docs=total_docs,
    )

    with open(os.path.join(OUTPUT_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)

    print("🎉 Dashboard updated successfully!")


if __name__ == "__main__":
    generate_dashboard()
