# ============================================
# 📄 generate_dashboard.py
# Version 5.5 – Adds theme analysis data
# ============================================

import os
import json
from jinja2 import Environment, FileSystemLoader
from LDA_engine_with_BERTopic_v054 import generate_topic_results

OUTPUT_DIR = "dashboard"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_dashboard():
    print("🚀 Generating dashboard...")

    docs, summaries, model, embeddings, theme_scores = generate_topic_results()

    # Save yesterday topic persistence
    with open(os.path.join(OUTPUT_DIR, "yesterday_topics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {str(k): {"embedding": embeddings[k], "title": summaries[k]["title"]} for k in embeddings},
            f,
            indent=2
        )

    # Format summaries
    summary_list = [
        {
            "topic_id": k,
            "title": v.get("title", ""),
            "summary": v.get("summary", "").replace("\n", "<br>"),
        }
        for k, v in summaries.items()
        if isinstance(v, dict)
    ]

    # Render to HTML
    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("dashboard_template.html")

    rendered_html = template.render(
        topic_map=model.visualize_topics().to_html(full_html=False),
        barchart=model.visualize_barchart(top_n_topics=5).to_html(full_html=False),
        summaries=summary_list,
        themes=theme_scores,
        run_date=os.getenv("RUN_DATE", "Today"),
    )

    with open(os.path.join(OUTPUT_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(rendered_html)

    print("🎉 Dashboard completed")

if __name__ == "__main__":
    generate_dashboard()
