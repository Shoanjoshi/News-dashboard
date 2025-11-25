# ============================================
# 📄 LDA_engine_with_BERTopic_v054.py
# Version 5.4 – Natural clustering + improved summaries + topic persistence support
# ============================================

import os
import feedparser
import re
from openai import OpenAI
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --------------------------------------------
# 1️⃣ RSS SOURCES (existing + new feeds)
# --------------------------------------------
RSS_FEEDS = [
    # EXISTING
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.reuters.com/reuters/markets",
    "https://www.ft.com/rss/home/us",
    "https://www.ft.com/rss/home/europe",
    "https://www.ft.com/rss/home/asia",
    "https://www.wsj.com/xml/rss/3_7014.xml",
    "https://www.wsj.com/xml/rss/3_7085.xml",
    "https://www.bloomberg.com/feeds/podcast/etf.xml",
    "https://www.bloomberg.com/markets/economics.rss",
    "https://www.bloomberg.com/feeds/bfm/podcast-odd-lots.xml",
    "https://feeds.marketwatch.com/marketwatch/topstories/",
    "https://feeds.marketwatch.com/marketwatch/marketpulse/",
    "https://feeds.reuters.com/reuters/worldNews",
    "https://feeds.reuters.com/reuters/politicsNews",
    "https://www.aljazeera.com/xml/rss/all.xml",
    "http://feeds.bbci.co.uk/news/world/rss.xml",
    "http://feeds.bbci.co.uk/news/politics/rss.xml",
    "http://rss.cnn.com/rss/edition_world.rss",
    "https://asia.nikkei.com/rss/feed",
    "https://economictimes.indiatimes.com/rssfeedsdefault.cms",
    "https://www.scmp.com/rss/91/feed",
    "https://www.euronews.com/rss?level=theme&name=business",
    "https://www.economist.com/europe/rss.xml",
    "https://www.theguardian.com/world/rss",
    "https://www.theguardian.com/business/rss",
    "https://feeds.reuters.com/reuters/technologyNews",
    "https://www.techspot.com/backend.xml",
    "https://feeds.feedburner.com/TechCrunch/",

    # NEW (systemic risk & leverage)
    "https://markets.businessinsider.com/rss",
    "https://www.investing.com/rss/news_25.rss",
    "https://www.investing.com/rss/news_1.rss",
    "https://www.investing.com/rss/news_285.rss",
    "https://www.federalreserve.gov/feeds/data.xml",
    "https://www.federalreserve.gov/feeds/press_all.xml",
    "https://www.bis.org/rss/press_rss.xml",
    "https://www.imf.org/external/np/exr/feeds/rss.aspx?type=imfnews",
    "https://www.risk.net/feeds/rss",
    "https://www.forbes.com/finance/feed",
    "https://feeds.feedburner.com/alternativeinvestmentnews",
    "https://www.eba.europa.eu/eba-news-rss",
]

# --------------------------------------------
# 2️⃣ Data Cleaning
# --------------------------------------------
def clean_text(text):
    text = re.sub(r"\s+", " ", str(text)).strip()
    text = re.sub(r"<[^>]+>", "", text)
    return text

def fetch_articles():
    articles = []
    for feed in RSS_FEEDS:
        try:
            parsed = feedparser.parse(feed)
            for entry in parsed.entries:
                if hasattr(entry, "summary") and hasattr(entry, "title"):
                    text = clean_text(entry.title + " " + entry.summary)
                    if len(text.split()) > 8:
                        articles.append(text)
        except:
            pass
    print(f"📰 Total collected articles: {len(articles)}")
    return articles if len(articles) >= 10 else []

# --------------------------------------------
# 3️⃣ OpenAI Topic Summaries (original prompt kept)
# --------------------------------------------
def summarize_topic_gpt(topic_id, words, docs):
    snippet_text = "\n".join(f"- {d[:200]}..." for d in docs[:3])
    prompt = (
        "You are a senior risk strategist at a global bank preparing a concise daily briefing. "
        "Analyze the topic using the key terms and excerpts. Focus on key drivers, likely impact "
        "on markets or geopolitical risk, and sentiment.\n\n"
        "STRICT FORMAT ONLY:\n"
        "TITLE: <3–5 WORDS, UPPERCASE>\n"
        "SUMMARY: <2–3 concise sentences>\n"
        f"Topic ID: {topic_id}\n"
        f"Key Terms: {', '.join(words[:10])}\n"
        f"Example Snippets:\n{snippet_text}\n"
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",      # Changed model fix
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=250
        )
        content = response.choices[0].message.content.strip()
        title, summary = None, None
        for line in content.split("\n"):
            if line.lower().startswith("title:"):
                title = line.split(":", 1)[1].strip()
            elif line.lower().startswith("summary:"):
                summary = line.split(":", 1)[1].strip()
        if not title:
            title = f"Topic {topic_id}"
        if not summary:
            summary = ", ".join(words[:5]) + " (fallback)"
        return {"title": title, "summary": summary}
    except Exception as e:
        print(f"⚠️ GPT error: {e}")
        return {"title": f"Topic {topic_id}", "summary": ", ".join(words[:5]) + " (fallback)"}

# --------------------------------------------
# 4️⃣ Topic Modeling
# --------------------------------------------
def run_bertopic_analysis(docs):
    umap_model = UMAP(n_neighbors=15, n_components=2, min_dist=0.1, metric="cosine", random_state=42)
    hdbscan_model = HDBSCAN(min_cluster_size=8, min_samples=1)
    vectorizer = CountVectorizer(stop_words="english")
    model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        nr_topics=None, top_n_words=15, verbose=True
    )
    topics, _ = model.fit_transform(docs)
    return model, topics

# --------------------------------------------
# 5️⃣ Main Runner
# --------------------------------------------
def generate_topic_results():
    docs = fetch_articles()
    if not docs:
        return [], {}, None

    model, topics = run_bertopic_analysis(docs)
    topic_info = model.get_topic_info()

    summary_dict = {}
    for topic_id in topic_info.Topic:
        if topic_id == -1:
            continue
        words = [t[0] for t in model.get_topic(topic_id)]
        docs_for_topic = [docs[i] for i, t in enumerate(topics) if t == topic_id]
        summary_dict[topic_id] = summarize_topic_gpt(topic_id, words, docs_for_topic) if docs_for_topic else "No documents"

    return docs, summary_dict, model

if __name__ == "__main__":
    docs, summaries, model = generate_topic_results()
    print(summaries)
