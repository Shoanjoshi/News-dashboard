# ============================================
# 📄 generate_dashboard.py
# Version 5.4 – Stable + topic summaries table
# ============================================

import os
from jinja2 import Environment, FileSystemLoader
from LDA_engine_with_BERTopic_v054 import generate_topic_results

OUTPUT_DIR = "dashboard"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_dashboard():
    print("🚀 Starting dashboard generation...")
    docs, topic_summaries, topic_model = generate_topic_results()

    if not docs or not topic_model:
        html = "<h3>⚠️ Not enough data to generate dashboard.</h3>"
        with open(os.path.join(OUTPUT_DIR, "index.html"), "w") as f:
            f.write(html)
        return

    # Visuals
    print("📊 Building visualizations...")
    try:
        fig_topics = topic_model.visualize_topics(width=600, height=650)
        html_topic_map = fig_topics.to_html(full_html=False)
    except:
        html_topic_map = "<p>No topic map</p>"

    try:
        fig_barchart = topic_model.visualize_barchart(top_n_topics=5)
        html_barchart = fig_barchart.to_html(full_html=False)
    except:
        html_barchart = "<p>No barchart</p>"

    # Convert summaries (now dicts)
    summary_list = [{"topic_id": k, "title": v["title"], "summary": v["summary"]} for k, v in topic_summaries.items()]

    template = Environment(loader=FileSystemLoader("templates")).get_template("dashboard_template.html")

    html = template.render(topic_map=html_topic_map, barchart=html_barchart, summaries=summary_list)

    with open(os.path.join(OUTPUT_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)

    print("🎉 Dashboard generated → dashboard/index.html")

if __name__ == "__main__":
    generate_dashboard()
