import os
import json
import numpy as np
from jinja2 import Environment, FileSystemLoader
import plotly.graph_objects as go

from LDA_engine_with_BERTopic_v054 import generate_topic_results, THEMES

OUTPUT_DIR = "dashboard"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOPIC_PERSISTENCE_JSON = os.path.join(OUTPUT_DIR, "yesterday_topics.json")
THEME_SIGNALS_JSON = os.path.join(OUTPUT_DIR, "yesterday_theme_signals.json")


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
            f, indent=2
        )


def _load_previous_theme_signals():
    if not os.path.exists(THEME_SIGNALS_JSON):
        return {}
    with open(THEME_SIGNALS_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_theme_signals(metrics):
    with open(THEME_SIGNALS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def _stretch(fig):
    fig.update_layout(autosize=True, margin=dict(l=5, r=5, t=40, b=5))
    return fig


def _build_theme_scatter(theme_signals, total_docs):
    themes = list(theme_signals.keys())
    xs = [theme_signals[t]["topicality"] for t in themes]
    ys = [theme_signals[t]["centrality"] for t in themes]

    fig = go.Figure(go.Scatter(
        x=xs, y=ys, mode="markers+text", text=themes, textposition="top center"
    ))
    fig.update_layout(
        title=f"Theme Distance Map – {total_docs} articles",
        xaxis_title="Topicality (% change)",
        yaxis_title="Centrality",
    )
    return fig.to_html(full_html=False)


def _build_heatmap(topic_summaries, theme_metrics, docs, topic_model):
    ordered_topics = sorted(topic_summaries.keys())
    row_labels = [topic_summaries[t]["title"] for t in ordered_topics]
    col_labels = THEMES + ["Others"]

    heat = np.zeros((len(ordered_topics), len(col_labels)), dtype=int)

    topics = topic_model.topics_

    for article_id, topic_id in enumerate(topics):
        if topic_id not in topic_summaries:
            continue
        row = ordered_topics.index(topic_id)

        for col, theme in enumerate(col_labels):
            if article_id in theme_metrics[theme]["articles_raw"]:
                heat[row, col] += 1

    fig = go.Figure(go.Heatmap(
        z=heat,
        x=col_labels,
        y=row_labels,
        colorscale="Blues",
    ))
    fig.update_layout(title="Topic × Theme Volume Heatmap")
    return fig.to_html(full_html=False)


def generate_dashboard():
    docs, topic_summaries, model, embeddings, theme_metrics = generate_topic_results()
    total_docs = len(docs)

    if not docs or not model:
        return

    _save_topic_embeddings(embeddings, topic_summaries)

    prev = _load_previous_theme_signals()

    for t, m in theme_metrics.items():
        prev_vol = prev.get(t, {}).get("volume", 0)
        m["topicality"] = (m["volume"] - prev_vol) / prev_vol if prev_vol else 0

    for metric in ["centrality", "topicality"]:
        ranked = sorted(theme_metrics.items(), key=lambda x: -x[1][metric])
        for rank, (theme, _) in enumerate(ranked, start=1):
            theme_metrics[theme][f"{metric}_rank"] = rank

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

    html_topic_map = _stretch(model.visualize_topics()).to_html(full_html=False)
    html_theme_map = _build_theme_scatter(theme_signals, total_docs)
    html_heatmap = _build_heatmap(topic_summaries, theme_metrics, docs, model)

    summaries = [
        {
            "topic_id": tid,
            "title": info["title"],
            "summary": info["summary"].replace("\n", "<br>"),
            "article_count": info["article_count"],
            "is_new": info.get("status") == "NEW",
            "is_persistent": info.get("status") == "PERSISTENT",
        }
        for tid, info in topic_summaries.items()
    ]

    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("dashboard_template.html")

    html = template.render(
        topic_map=html_topic_map,
        theme_map=html_theme_map,
        heatmap=html_heatmap,
        theme_signals=theme_signals,
        summaries=summaries,
        run_date=os.getenv("RUN_DATE", "Today"),
        total_docs=total_docs,
    )

    with open(os.path.join(OUTPUT_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)

    print("Dashboard updated.")


if __name__ == "__main__":
    generate_dashboard()
