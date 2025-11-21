
# ============================================
# 📄 generate_dashboard.py
# Version 5.3 – Stable dashboard with GPT-5-nano & fallback visualizations
# ============================================

import os
import json
from jinja2 import Environment, FileSystemLoader

from LDA_engine_with_BERTopic_v054 import generate_topic_results

# Ensure OpenAI key exists
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("⚠️ OPENAI_API_KEY not found. Add it as a GitHub Secret.")

# Output directory for dashboard export
OUTPUT_DIR = "dashboard"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_dashboard():
    print("🚀 Starting dashboard generation...")

    # 1️⃣ Run topic modeling and summarization
    docs, topic_summaries, topic_model = generate_topic_results()

    # If no documents or no model → fallback simple dashboard
    if not docs or not topic_model:
        print("⚠️ Not enough data for full dashboard. Using fallback layout.")
        html_content = "<h3>No sufficient data to generate dashboard today.</h3>"
        with open(os.path.join(OUTPUT_DIR, "index.html"), "w", encoding="utf-8") as f:
            f.write(html_content)
        print("🟡 Dashboard generated with fallback content.")
        return

    # 2️⃣ Prepare topic visualization
    print("📊 Building visualizations...")

    try:
        # Plot topic distance map (bubble chart)
        fig_topics = topic_model.visualize_topics(width=600, height=650)
    except Exception as e:
        print(f"⚠️ Unable to build topic map. Reason: {e}")
        fig_topics = None

    try:
        # Barchart: keyword distributions
        fig_barchart = topic_model.visualize_barchart(top_n_topics=5)
    except Exception as e:
        print(f"⚠️ Unable to build barchart. Reason: {e}")
        fig_barchart = None

    # 3️⃣ Convert graphs to HTML
    html_topic_map = fig_topics.to_html(full_html=False) if fig_topics else "<p>No topic map available.</p>"
    html_barchart = fig_barchart.to_html(full_html=False) if fig_barchart else "<p>No bar chart available.</p>"

    # 4️⃣ Export topic summaries
    summary_list = [
        {"topic_id": k, "summary": v.replace("\n", "<br>")}
        for k, v in topic_summaries.items()
    ]

    # 5️⃣ Render dashboard via Jinja2 template
    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("dashboard_template.html")

    rendered_html = template.render(
        topic_map=html_topic_map,
        barchart=html_barchart,
        summaries=summary_list,
    )

    # 6️⃣ Write to HTML
    output_path = os.path.join(OUTPUT_DIR, "index.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(rendered_html)

    print(f"🎉 Dashboard successfully generated → {output_path}")


# --------------------------------------------
# Run when executed manually
# --------------------------------------------
if __name__ == "__main__":
    generate_dashboard()
