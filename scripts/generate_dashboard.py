# ============================================
# 📄 generate_dashboard.py
# Version 5.5 – Persistence tracking + executive styling + fallback-safe
# ============================================

import os
import json
from datetime import datetime
import numpy as np # ✅ Moved here (good practice)

from jinja2 import Environment, FileSystemLoader
from LDA_engine_with_BERTopic_v054 import generate_topic_results

# Ensure OpenAI key exists
if not os.getenv("OPENAI_API_KEY"):
  raise ValueError("⚠️ OPENAI_API_KEY not found. Add it as a GitHub Secret.")

# Output directory for dashboard export
OUTPUT_DIR = "dashboard"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Location to store yesterday’s topic JSON state
YESTERDAY_FILE = os.path.join(OUTPUT_DIR, "yesterday_topics.json")


def generate_dashboard():
print("🚀 Starting dashboard generation...")

# 🔹 Accept 4 outputs (added persistence tracking)
docs, topic_summaries, topic_model, embeddings = generate_topic_results()

if not docs or not topic_model:
print("⚠️ Not enough data for full dashboard. Using fallback layout.")
with open(os.path.join(OUTPUT_DIR, "index.html"), "w", encoding="utf-8") as f:
f.write("<h3>No sufficient data to generate dashboard today.</h3>")
return

# 🔍 Load yesterday’s topics if available
if os.path.exists(YESTERDAY_FILE):
with open(YESTERDAY_FILE, "r") as f:
yesterday_data = json.load(f)
else:
yesterday_data = {}
print("🔸 No previous topic data found. First run → all topics marked as new.")

# 🔁 Topic persistence detection
for topic_id, data in topic_summaries.items():
emb_today = embeddings.get(topic_id)

if not emb_today or not yesterday_data:
data["is_new"] = True
data["is_persistent"] = False
continue

best_match_sim = 0.0
for _, old_data in yesterday_data.items():
try:
sim = np.dot(emb_today, old_data["embedding"]) / (
np.linalg.norm(emb_today) * np.linalg.norm(old_data["embedding"])
)
best_match_sim = max(best_match_sim, sim)
except:
continue

data["is_persistent"] = best_match_sim > 0.75
data["is_new"] = not data["is_persistent"]

# 🧠 Plot visualizations
print("📊 Building visualizations...")
try:
fig_topics = topic_model.visualize_topics(width=600, height=650)
except Exception:
fig_topics = None

try:
fig_barchart = topic_model.visualize_barchart(top_n_topics=5)
except Exception:
fig_barchart = None

html_topic_map = fig_topics.to_html(full_html=False) if fig_topics else "<p>No topic map available.</p>"
html_barchart = fig_barchart.to_html(full_html=False) if fig_barchart else "<p>No bar chart available.</p>"

# 📝 Prepare summaries for dashboard
summary_list = [
{
"topic_id": k,
"title": v.get("title", ""),
"summary": v.get("summary", "").replace("\n", "<br>"),
"is_new": v.get("is_new", False),
"is_persistent": v.get("is_persistent", False),
}
for k, v in topic_summaries.items()
if isinstance(v, dict)
]

# 🧾 Save JSON state for the next comparison
tracking_json = {
topic_id: {"embedding": embeddings.get(topic_id, [])}
for topic_id in topic_summaries.keys()
}
with open(YESTERDAY_FILE, "w") as f:
json.dump(tracking_json, f, indent=2)

# 📄 Render HTML dashboard
env = Environment(loader=FileSystemLoader("templates"))
template = env.get_template("dashboard_template.html")
rendered_html = template.render(
topic_map=html_topic_map,
barchart=html_barchart,
summaries=summary_list,
run_date=datetime.utcnow().strftime("%Y-%m-%d")
)

output_path = os.path.join(OUTPUT_DIR, "index.html")
with open(output_path, "w", encoding="utf-8") as f:
f.write(rendered_html)

print(f"🎉 Dashboard successfully generated → {output_path}")


# --------------------------------------------
# Run manually
# --------------------------------------------
if __name__ == "__main__":
generate_dashboard()
