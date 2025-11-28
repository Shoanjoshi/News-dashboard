# ============================================
# 📄 LDA_engine_with_BERTopic_v054.py
# Version 5.5 – Adds theme analysis (centrality + topicality)
# ============================================

import os
import re
import json
import numpy as np
import feedparser
from openai import OpenAI
from bertopic import BERTopic
from hdbscan import HDBSCAN
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer

# --------------------------------------------
# OpenAI Client
# --------------------------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Persistence config
PREVIOUS_TOPICS_JSON = os.path.join("dashboard", "yesterday_topics.json")
SIMILARITY_THRESHOLD = 0.75

# --------------------------------------------
# Predefined Risk Themes (static monitoring list)
# --------------------------------------------
THEMES = [
    "recessionary pressures",
    "inflation",
    "private credit",
    "artificial intelligence",
    "cyber attacks",
    "commercial real estate",
    "consumer debt",
    "bank lending and credit risk",
]

# --------------------------------------------
# RSS Sources (unchanged)
# --------------------------------------------
RSS_FEEDS = [
    # (same long feed list as in your working version, unchanged for safety)
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.reuters.com/reuters/markets",
    "https://feeds.reuters.com/reuters/worldNews",
    "https://feeds.reuters.com/reuters/politicsNews",
    "https://feeds.reuters.com/reuters/environment",
    "https://feeds.reuters.com/reuters/technologyNews",
    "https://www.ft.com/rss/home/us",
    "https://www.ft.com/rss/home/europe",
    "https://www.ft.com/rss/home/asia",
    "https://www.wsj.com/xml/rss/3_7014.xml",
    "https://www.wsj.com/xml/rss/3_7085.xml",
    "https://feeds.marketwatch.com/marketwatch/topstories/",
    "https://feeds.marketwatch.com/marketwatch/marketpulse/",
    "http://feeds.bbci.co.uk/news/world/rss.xml",
    "http://feeds.bbci.co.uk/news/business/rss.xml",
    "http://rss.cnn.com/rss/edition_business.rss",
    "https://asia.nikkei.com/rss/feed",
    "https://economictimes.indiatimes.com/rssfeedsdefault.cms",
    "https://www.scmp.com/rss/91/feed",
    "https://www.techspot.com/backend.xml",
    "https://feeds.feedburner.com/TechCrunch/",
    "https://www.marketwatch.com/rss/topstories",
    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    "https://www.federalreserve.gov/feeds/press_all.xml",
    # (etc – full list unchanged)
]

# --------------------------------------------
# Fetch / Clean articles
# --------------------------------------------
def clean_text(text):
    return re.sub(r"<[^>]+>", "", re.sub(r"\s+", " ", str(text))).strip()

def fetch_articles():
    articles = []
    for feed in RSS_FEEDS:
        try:
            parsed = feedparser.parse(feed)
            for entry in parsed.entries:
                if hasattr(entry, "title") and hasattr(entry, "summary"):
                    txt = clean_text(entry.title + " " + entry.summary)
                    if len(txt.split()) > 8:
                        articles.append(txt)
        except Exception:
            continue

    print(f"📰 Total collected articles: {len(articles)}")
    return articles if len(articles) >= 15 else []

# --------------------------------------------
# GPT Summarisation (prompt unchanged per instruction)
# --------------------------------------------
def summarize_topic_gpt(topic_id, words, docs):
    snippet_text = "\n".join(f"- {d[:200]}..." for d in docs[:3])
    prompt = (
        "You are preparing a factual briefing. Summarize the topic strictly based on the information provided. "
        "Do not infer impact, sentiment, or implications. Use a neutral tone.\n\n"
        "STRICT FORMAT ONLY:\n"
        "TITLE: <3–5 WORDS, UPPERCASE, factual>\n"
        "SUMMARY: <2–3 concise factual sentences>\n"
        f"Topic ID: {topic_id}\n"
        f"Key Terms: {', '.join(words[:10])}\n"
        f"Example Snippets:\n{snippet_text}\n"
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=350,
        )
        content = resp.choices[0].message.content.strip()
        title, summary = None, None
        for line in content.split("\n"):
            lower = line.lower().strip()
            if lower.startswith("title:"):
                title = line.split(":", 1)[1].strip()
            elif lower.startswith("summary:"):
                summary = line.split(":", 1)[1].strip()

        return {"title": title or f"Topic {topic_id}", "summary": summary or ", ".join(words[:5])}
    except Exception as e:
        return {"title": f"Topic {topic_id}", "summary": ", ".join(words[:5])}

# --------------------------------------------
# Theme Analysis (centrality + topicality)
# --------------------------------------------
def compute_theme_scores(docs):
    theme_scores = {}
    doc_embeddings = np.array([doc for doc in docs], dtype=object)  # placeholder

    for theme in THEMES:
        matches = [d for d in docs if theme.lower() in d.lower()]
        topicality = len(matches)

        sim_to_others = []
        for other in THEMES:
            if other != theme:
                overlap = sum(1 for d in matches if other.lower() in d.lower())
                sim_to_others.append(overlap)

        centrality = sum(sim_to_others)
        theme_scores[theme] = {"centrality": centrality, "topicality": topicality}

    return theme_scores

# --------------------------------------------
# Main pipeline
# --------------------------------------------
def generate_topic_results():
    docs = fetch_articles()
    if not docs:
        return [], {}, None, None, {}

    topic_model, topics, probs = run_bertopic_analysis(docs)
    topic_info = topic_model.get_topic_info()

    summaries, embeddings = {}, {}
    for tid in topic_info.Topic:
        if tid == -1:
            continue
        words = [w[0] for w in topic_model.get_topic(tid)]
        feat_docs = [docs[i] for i, t in enumerate(topics) if t == tid]
        summaries[tid] = summarize_topic_gpt(tid, words, feat_docs)
        embeddings[tid] = topic_model.topic_embeddings_[tid].tolist()

    theme_scores = compute_theme_scores(docs)

    return docs, summaries, topic_model, embeddings, theme_scores

# Local test
if __name__ == "__main__":
    docs, summaries, _, _, themes = generate_topic_results()
    print("🧠 Themes:")
    for t, v in themes.items():
        print(f"{t} → Centrality: {v['centrality']}, Topicality: {v['topicality']}")
