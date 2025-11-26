# ============================================
# 📄 generate_dashboard.py
# Version 5.4 – Supports topic persistence tracking
# ============================================

import os
import json  # 🔹 Needed for saving JSON file
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

    # 🔹 Accept 4 outputs (added persistence tracking)
    docs, topic_summaries, topic_model, embeddings = generate_topic_results()

    if not docs or not topic_model:
        print("⚠️ Not enough data for full dashboard. Using fallback layout.")
        html_content = "<h3>No sufficient data to generate dashboard today.</h3>"
        with open(os.path.join(OUTPUT_DIR, "index.html"), "w", encoding="utf-8") as f:
            f.write(html_content)
        print("🟡 Dashboard generated with fallback content.")
        return

    # 🔹 NEW — Save embeddings & titles for persistence tracking
    json_path = os.path.join(OUTPUT_DIR, "yesterday_topics.json")
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {str(k): {"embedding": embeddings[k], "title": topic_summaries[k].get("title", "")}
                 for k in embeddings.keys()},
                f,
                indent=2
            )
        print(f"📁 Saved topic persistence file → {json_path}")
    except Exception as e:
        print(f"❌ ERROR saving JSON file: {e}")

    print("📊 Building visualizations...")

    try:
        fig_topics = topic_model.visualize_topics(width=600, height=650)
    except Exception as e:
        print(f"⚠️ Unable to build topic map. Reason: {e}")
        fig_topics = None

    try:
        fig_barchart = topic_model.visualize_barchart(top_n_topics=5)  
