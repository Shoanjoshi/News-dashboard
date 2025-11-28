# ============================================
# 📄 generate_dashboard.py
# Version 5.5 – Stable with persistence tracking
# ============================================

import os
import json
from jinja2 import Environment, FileSystemLoader

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

def generate_dashboard():
    print("🚀 Generating dashboard...")

    # Expect 5 outputs (last one placeholder for future themes)
    docs, topic_summaries, topic_model, embeddings, _ = generate_topic_results()

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
    json_path = os.path.join(OUTPUT_DIR, "yesterday_topics.json")
    try:
        data_to_save = {
            str(k): {
                "embedding": embeddings[k],
                "title": topic_summaries[k].get("title", ""),
            }
            for k in embeddings.keys()
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, indent=2)
        print(f"📁 Saved topic persistence file → {json_path}")
    except Exception as e:
        print(f"❌ ERROR saving topic JSON file: {e}")

    # --------------------------------------------
    # 3️⃣ Build Visualizations
    # --------------------------------------------
    try:
        fig_topics = topic_model.visualize_topics(width=600, height=650)
        html_topic_map = fig_topics.to_html(full_html=False)
    except Exception as e:
        print(f"⚠️ Error generating topic map: {e}")
        html_topic_map = "<p>No topic map available.</p>"

    try:
        fig_barchart = topic_model.visualize_barchart(top_n_topics=5)
        html_barchart = fig_barchart.to_html(full_html=False)
    except Exception as e:
        print(f"⚠️ Error generating barchart: {e}")
        html_barchart = "<p>No bar chart available.</p>"

    # --------------------------------------------
    # 4️⃣ Formatting data for template
    # --------------------------------------------
    summary_list = []
    for k, v in topic_summaries.items():
        if isinstance(v, dict):
            summary_list.append({
                "topic_id": k,
                "title": v.get("title", ""),
                "summary": v.get("summary", "").replace("\n", "<br>"),
                "is_new": v.get("status", "").upper() == "NEW",
                "is_persistent": v.get("status", "").upper() == "PERSISTENT",
            })

    # --------------------------------------------
    # 5️⃣ Render via Jinja2
    # --------------------------------------------
    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("dashboard_template.html")

    rendered_html = template.render(
        topic_map=html_topic_map,
        barchart=html_barchart,
        summaries=summary_list,
        run_date=os.getenv("RUN_DATE", "Today")
    )

    # --------------------------------------------
    # 6️⃣ Save HTML
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
