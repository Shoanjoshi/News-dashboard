# ================================================
# 🔍 LDA_engine_with_BERTopic_v054 (Updated)
# Version 5.8 – Includes semantic theme assignment
# ================================================
import os
import json
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import feedparser
import openai

# 🔒 Set API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# ======================================================
# 1️⃣ SETTINGS
# ======================================================
RSS_FEEDS = [
    # Your existing RSS feeds…
]

THEMES = [
    "Recessionary pressures",
    "Inflation",
    "Private credit",
    "AI",
    "Cyber attacks",
    "Commercial real estate",
    "Consumer debt",
    "Bank lending and credit risk",
]
SIMILARITY_THRESHOLD = 0.5  # minimum theme confidence to assign
PROMPT = """
You are preparing a factual briefing. Summarize the topic strictly based on the information provided.
Do not infer impact, sentiment, or implications. Avoid subjective language, predictions, or assumptions.
Use neutral, objective tone.

STRICT FORMAT ONLY:
TITLE: <3–5 WORDS, UPPERCASE, factual>
SUMMARY: <2–3 concise factual sentences. No speculation.>
"""


# ======================================================
# 2️⃣ Fetch Articles
# ======================================================
def fetch_rss_articles():
    docs = []
    for feed in RSS_FEEDS:
        parsed = feedparser.parse(feed)
        for entry in parsed.entries[:20]:
            content = entry.get("summary", "")[:800]
            if content:
                docs.append(content)
    return docs


# ======================================================
# 3️⃣ Summarization via GPT
# ======================================================
def summarize_topic(text):
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": PROMPT + "\n" + text}]
    )
    return response.choices[0].message.content


# ======================================================
# 4️⃣ Theme Embedding for Semantic Classification
# ======================================================
def load_embedding_model(topic_model):
    """Use the same embedding model as BERTopic if set; else load default."""
    if hasattr(topic_model, "embedding_model") and topic_model.embedding_model:
        print("📌 Using BERTopic’s embedding model for theme matching.")
        return topic_model.embedding_model
    else:
        print("⚠️ No embedding model attached to BERTopic. Using MiniLM.")
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def get_theme_embeddings(embedding_model):
    return embedding_model.encode(THEMES, show_progress_bar=False)


def assign_theme_to_topic(topic_embedding, theme_embeddings):
    similarities = cosine_similarity([topic_embedding], theme_embeddings)[0]
    best_index = np.argmax(similarities)
    best_score = similarities[best_index]
    if best_score >= SIMILARITY_THRESHOLD:
        return THEMES[best_index], best_score
    return "Others", best_score


# ======================================================
# 5️⃣ Main Topic Generation & Theme Attribution Logic
# ======================================================
def generate_topic_results():
    print("🚀 Fetching news articles…")
    docs = fetch_rss_articles()
    if not docs:
        return [], {}, None, {}, {}

    print(f"📊 {len(docs)} articles fetched.")

    # === Generate BERTopic Clusters ===
    topic_model = BERTopic(nr_topics="auto")
    topics, probabilities = topic_model.fit_transform(docs)
    embeddings = topic_model._embedding_model.encode(docs, show_progress_bar=False)

    # === Apply GPT Summaries ===
    topic_summaries = {}
    for topic_id in sorted(set(topics)):
        representative_docs = [docs[i] for i, t in enumerate(topics) if t == topic_id][:3]
        combined_text = " ".join(representative_docs)
        summary_text = summarize_topic(combined_text)
        topic_summaries[topic_id] = {
            "title": summary_text.split("SUMMARY:")[0].replace("TITLE:", "").strip(),
            "summary": summary_text.split("SUMMARY:")[-1].strip(),
            "status": "NEW"  # updated later
        }

    # === Topic Persistence ===
    previous_topics_path = "dashboard/yesterday_topics.json"
    if os.path.exists(previous_topics_path):
        with open(previous_topics_path, "r") as f:
            previous = json.load(f)
        for topic_id, summary in topic_summaries.items():
            match = "PERSISTENT" if str(topic_id) in previous else "NEW"
            summary["status"] = match

    # === Semantic Theme Classification ===
    print("🧠 Assigning themes using embedding similarity...")
    embedding_model = load_embedding_model(topic_model)
    theme_embeddings = get_theme_embeddings(embedding_model)

    theme_scores = {theme: {"volume": 0, "centrality": 0} for theme in THEMES}
    theme_scores["Others"] = {"volume": 0, "centrality": 0}

    for i, topic_id in enumerate(topics):
        topic_embedding = embeddings[i]
        assigned_theme, _ = assign_theme_to_topic(topic_embedding, theme_embeddings)
        theme_scores[assigned_theme]["volume"] += 1

    print("📌 Theme allocation complete.")

    return docs, topic_summaries, topic_model, embeddings, theme_scores


# ======================================================
# 🧪 Debug Run
# ======================================================
if __name__ == "__main__":
    print("🧪 Running in debug mode...")
    results = generate_topic_results()
    print(json.dumps(results[4], indent=2))
