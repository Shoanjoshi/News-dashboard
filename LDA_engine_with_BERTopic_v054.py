# ============================================
# 📄 LDA_engine_with_BERTopic_v054.py
# Version 5.4 – Natural topic clustering & diversified RSS sources
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
# 2️⃣ Optimized & Diversified RSS Sources
# --------------------------------------------
RSS_FEEDS = [
    # 📈 Global Business & Economy
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.reuters.com/reuters/markets",
    "https://www.ft.com/rss/home/us",
    "https://www.ft.com/rss/home/europe",
    "https://www.ft.com/rss/home/asia",
    "https://www.wsj.com/xml/rss/3_7014.xml",
    "https://www.wsj.com/xml/rss/3_7085.xml",

    # 💰 Banking, Private Credit, Finance
    "https://www.bloomberg.com/feeds/podcast/etf.xml",
    "https://www.bloomberg.com/markets/economics.rss",
    "https://www.bloomberg.com/feeds/bfm/podcast-odd-lots.xml",
    "https://feeds.marketwatch.com/marketwatch/topstories/",
    "https://feeds.marketwatch.com/marketwatch/marketpulse/",

    # 🌍 World Politics & Geopolitics
    "https://feeds.reuters.com/reuters/worldNews",
    "https://feeds.reuters.com/reuters/politicsNews",
    "https://www.aljazeera.com/xml/rss/all.xml",
    "http://feeds.bbci.co.uk/news/world/rss.xml",
    "http://feeds.bbci.co.uk/news/politics/rss.xml",
    "http://rss.cnn.com/rss/edition_world.rss",

    # 🏦 Asia & Emerging Markets
    "https://asia.nikkei.com/rss/feed",
    "https://economictimes.indiatimes.com/rssfeedsdefault.cms",
    "https://www.scmp.com/rss/91/feed",

    # 🇪🇺 Europe Economy
    "https://www.euronews.com/rss?level=theme&name=business",
    "https://www.economist.com/europe/rss.xml",
    "https://www.theguardian.com/world/rss",
    "https://www.theguardian.com/business/rss",

    # 🚀 Tech & Innovation
    "https://feeds.reuters.com/reuters/technologyNews",
    "https://www.techspot.com/backend.xml",
    "https://feeds.feedburner.com/TechCrunch/",
]

# --------------------------------------------
# 3️⃣ Fetch & Clean Articles
# --------------------------------------------
def clean_text(text):
    text = re.sub(r'\s+', ' ', str(text)).strip()
    text = re.sub(r'<[^>]+>', '', str(text))
    return text

def fetch_articles():
    articles = []
    for feed in RSS_FEEDS:
        try:
            parsed_feed = feedparser.parse(feed)
            for entry in parsed_feed.entries:
                if hasattr(entry, "summary") and hasattr(entry, "title"):
                    clean_article = clean_text(entry.title + " " + entry.summary)
                    if len(clean_article.split()) > 8:
                        articles.append(clean_article)
        except Exception:
            continue

    print(f"📰 Total collected articles: {len(articles)}")
    if len(articles) < 10:
        print("⚠️ Not enough valid articles. Skipping BERTopic.")
        return []
    return articles

# --------------------------------------------
# 4️⃣ GPT Summarization (Improved Logic)
# --------------------------------------------
def summarize_topic_gpt(topic_id, words, docs):
    snippet_text = "\n".join(f"- {d[:200]}..." for d in docs[:3])

    prompt = (
        "You are a senior risk strategist at a global bank preparing a concise daily briefing. "
        "Analyze the topic using the key terms and excerpts. Focus on key drivers and relevance to markets.\n\n"
        "STRICT FORMAT ONLY:\n"
        "TITLE: <3-5 WORDS, UPPERCASE>\n"
        "SUMMARY: <2-3 concise sentences>\n"
        f"Topic ID: {topic_id}\n"
        f"Key Terms: {', '.join(words[:10])}\n"
        f"Example Snippets:\n{snippet_text}\n"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=350,
            temperature=0.35
        )
        content = response.choices[0].message.content.strip()

        # --- NEW ROBUST PARSING ---
        title, summary = None, None
        for line in content.split("\n"):
            cleaned = line.strip().lower()
            if "title" in cleaned:
                title = line.split(":", 1)[-1].strip()
            elif "summary" in cleaned:
                summary = line.split(":", 1)[-1].strip()

        if not title:
            title = f"Topic {topic_id}"
        if not summary:
            summary = f"{', '.join(words[:5])} (fallback)"

        return {"title": title, "summary": summary}

    except Exception as e:
        print(f"⚠️ GPT error (fallback): {e}")
        return {"title": f"Topic {topic_id}", "summary": f"{', '.join(words[:5])} (fallback)"}


# --------------------------------------------
# 5️⃣ BERTopic Engine (Natural Topic Clustering)
# --------------------------------------------
def run_bertopic_analysis(docs):
    print("🚀 Running BERTopic using natural clustering...")

    umap_model = UMAP(
        n_neighbors=15,
        n_components=2,
        min_dist=0.1,
        metric="cosine",
        random_state=42
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=8,
        min_samples=1
    )

    vectorizer_model = CountVectorizer(stop_words="english")
    
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        nr_topics=None,
        top_n_words=15,
        vectorizer_model=vectorizer_model, 
        verbose=True,
    )

    topics, probs = topic_model.fit_transform(docs)
    return topic_model, topics, probs

# --------------------------------------------
# 6️⃣ Main Runner
# --------------------------------------------
def generate_topic_results():
    print("📡 Fetching articles...")
    docs = fetch_articles()
    if not docs:
        return [], {}, None

    topic_model, topics, probs = run_bertopic_analysis(docs)

    summary_dict = {}
    topic_info = topic_model.get_topic_info()

    if topic_info.shape[0] < 2:
        print("⚠️ Not enough topics detected! Fallback...")
        summary_dict[0] = "Not enough data for topic analysis"
        return docs, summary_dict, topic_model

    print(f"🧠 Detected {topic_info.shape[0]} topics.")

    for topic_id in topic_info.Topic:
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
# 7️⃣ Test Only
# --------------------------------------------
if __name__ == "__main__":
    docs, summaries, model = generate_topic_results()
    print("📊 Topic Summaries:\n")
    for k, v in summaries.items():
        print(f"🟢 {k}: {v}\n")









