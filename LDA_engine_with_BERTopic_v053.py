# ============================================
# 📄 LDA_engine_with_BERTopic_v053.py
# Version 5.3 – Optimized with GPT-5-nano & stable topic control
# ============================================

import os
import feedparser
import requests
import re
from tqdm import tqdm
from openai import OpenAI

from bertopic import BERTopic
from hdbscan import HDBSCAN
from umap import UMAP

from sklearn.feature_extraction.text import CountVectorizer

# --------------------------------------------
# 1️⃣ OpenAI Client
# --------------------------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --------------------------------------------
# 2️⃣ RSS Sources
# --------------------------------------------
RSS_FEEDS = [
    "http://feeds.bbci.co.uk/news/rss.xml",
    "https://rss.cnn.com/rss/edition.rss",
    "https://feeds.a.dj.com/rss/RSSWorldNews.xml",
    "https://www.reutersagency.com/feed/?best-sectors=world&post_type=best"
]

# --------------------------------------------
# 3️⃣ Fetch & Clean Articles
# --------------------------------------------
def clean_text(text):
    text = re.sub(r'\s+', ' ', str(text)).strip()
    text = re.sub(r'<[^>]+>', '', str(text))  # remove HTML
    return text

def fetch_articles():
    articles = []
    for feed in RSS_FEEDS:
        try:
            parsed_feed = feedparser.parse(feed)
            for entry in parsed_feed.entries:
                if hasattr(entry, "summary") and hasattr(entry, "title"):
                    clean_article = clean_text(entry.title + " " + entry.summary)
                    if len(clean_article.split()) > 5:
                        articles.append(clean_article)
        except Exception:
            continue

    if len(articles) < 8:
        print("⚠️ Not enough valid articles. Skipping BERTopic…")
        return []
    return articles

# --------------------------------------------
# 4️⃣ GPT-5-nano Summary
# --------------------------------------------
def summarize_topic_gpt(topic_id, words, docs):
    prompt = (
        f"You are an expert news analyst.\n"
        f"• Topic ID: {topic_id}\n"
        f"• Key Terms: {', '.join(words[:10])}\n"
        f"• Article Snippets:\n"
        + "\n".join(f"- {d[:200]}..." for d in docs[:3])
        + "\n\nProvide:\n- A concise summary (1-2 sentences)\n"
          "- A topic title (max 5 words, uppercase)."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=350,
            temperature=0.35
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"⚠️ GPT error, fallback: {e}")
        return f"Topic {topic_id}: {', '.join(words[:5])} (fallback)"

# --------------------------------------------
# 5️⃣ BERTopic Engine (Force 5 Topics)
# --------------------------------------------
def run_bertopic_analysis(docs):
    print("Running BERTopic…")

    umap_model = UMAP(
        n_neighbors=15,
        n_components=2,
        min_dist=0.1,
        metric="cosine",
        random_state=42
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=2,
        min_samples=1
    )

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        nr_topics=5,  # Force 5 topics
        top_n_words=15,
        verbose=True,
    )

    topics, probs = topic_model.fit_transform(docs)
    return topic_model, topics, probs

# --------------------------------------------
# 6️⃣ Main runner
# --------------------------------------------
def generate_topic_results():
    print("Fetching articles…")
    docs = fetch_articles()
    if not docs:
        return [], {}, None

    topic_model, topics, probs = run_bertopic_analysis(docs)

    summary_dict = {}
    topic_info = topic_model.get_topic_info()

    if topic_info.shape[0] < 5:
        print("⚠️ Not enough usable topics after clustering. Fallback summarization.")
        for i in range(5):
            summary_dict[i] = f"Topic {i}: Insufficient data"
        return docs, summary_dict, topic_model

    for topic_id in topic_info.Topic.head():
        if topic_id == -1:
            continue
        words = [t[0] for t in topic_model.get_topic(topic_id)]
        feat_docs = [docs[idx] for idx, t in enumerate(topics) if t == topic_id]

        if feat_docs:
            summary_dict[topic_id] = summarize_topic_gpt(topic_id, words, feat_docs)
        else:
            summary_dict[topic_id] = "No documents for this topic."

    return docs, summary_dict, topic_model

# --------------------------------------------
# 7️⃣ For testing locally
# --------------------------------------------
if __name__ == "__main__":
    docs, summaries, model = generate_topic_results()
    print("📊 Topic Summaries:\n")
    for k, v in summaries.items():
        print(f"🟢 {k}: {v}\n")
