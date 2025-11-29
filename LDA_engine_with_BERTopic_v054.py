# ======================================================
# LDA_engine_with_BERTopic_v054.py (Restored & Stabilized)
# Version 5.9 – Includes RSS restoration + safety checks
# ======================================================

import os
import json
import feedparser
import openai
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

openai.api_key = os.getenv("OPENAI_API_KEY")

# --------------------------------------------
# 🔄 RESTORED FULL RSS FEED LIST (as provided)
# --------------------------------------------
RSS_FEEDS = [
    # US Economic & Business
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.reuters.com/reuters/markets",
    "https://www.ft.com/rss/home/us",
    "https://www.wsj.com/xml/rss/3_7014.xml",
    "https://www.wsj.com/xml/rss/3_7085.xml",
    "https://feeds.marketwatch.com/marketwatch/topstories/",
    "https://feeds.marketwatch.com/marketwatch/marketpulse/",
    "http://feeds.bbci.co.uk/news/business/rss.xml",
    "http://rss.cnn.com/rss/edition_business.rss",

    # Europe & Asia
    "https://www.ft.com/rss/home/europe",
    "https://www.ft.com/rss/home/asia",
    "https://asia.nikkei.com/rss/feed",
    "https://www.scmp.com/rss/91/feed",

    # Technology
    "https://feeds.reuters.com/reuters/technologyNews",
    "https://feeds.feedburner.com/TechCrunch/",

    # New systemic risk / leverage / regulation feeds
    "https://www.ft.com/rss/home",
    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    "https://www.investing.com/rss/news_25.rss",
    "https://www.investing.com/rss/news_1.rss",
    "https://www.investing.com/rss/news_285.rss",
    "https://www.federalreserve.gov/feeds/data.xml",
    "https://www.federalreserve.gov/feeds/press_all.xml",
    "https://markets.businessinsider.com/rss",
    "https://www.risk.net/feeds/rss",
    "https://www.forbes.com/finance/feed",
    "https://feeds.feedburner.com/alternativeinvestmentnews",
    "https://www.eba.europa.eu/eba-news-rss",
    "https://www.bis.org/rss/press_rss.xml",
    "https://www.imf.org/external/np/exr/feeds/rss.aspx?type=imfnews",
]

# --------------------------------------------
# Defined Themes
# --------------------------------------------
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

SIMILARITY_THRESHOLD = 0.5

PROMPT = """You are preparing a factual briefing. Summarize the topic strictly based on the information provided.
Do not infer impact, sentiment, or implications. Avoid subjective language, predictions, or assumptions.
Use neutral, objective tone.

STRICT FORMAT ONLY:
TITLE: <3–5 WORDS, UPPERCASE, factual>
SUMMARY: <2–3 concise factual sentences. No speculation.>"""


# ------------------------------------------------------
# 1️⃣ RSS Fetch Logic with Safety & Logging
# ------------------------------------------------------
def fetch_rss_articles():
    docs = []
    for feed in RSS_FEEDS:
        try:
            parsed = feedparser.parse(feed)
            for entry in parsed.entries[:20]:
                content = entry.get("summary") or entry.get("description") or entry.get("title") or ""
                if isinstance(content, str) and len(content.strip()) > 50:
                    docs.append(content.strip()[:1000])
        except Exception as e:
            print(f"⚠ Error reading feed {feed}: {e}")

    print(f"🧪 Total RSS articles extracted: {len(docs)}")
    if docs:
        print(f"📌 Sample article: {docs[0][:200]}")
    else:
        print("❌ No usable RSS content extracted!")
    return docs


# ------------------------------------------------------
# GPT topic summarization
# ------------------------------------------------------
def summarize_topic(text):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": PROMPT + "\n" + text}],
        )
        return response.choices[0].message.content
    except Exception:
        return "TITLE: UNKNOWN\nSUMMARY: Summary failed."


# ------------------------------------------------------
# Embedding logic
# ------------------------------------------------------
def load_embedding_model(topic_model):
    if hasattr(topic_model, "embedding_model") and topic_model.embedding_model:
        return topic_model.embedding_model
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def get_theme_embeddings(model):
    return model.encode(THEMES, show_progress_bar=False)


def assign_theme(topic_embedding, theme_embeddings):
    sims = cosine_similarity([topic_embedding], theme_embeddings)[0]
    best_idx = np.argmax(sims)
    if sims[best_idx] >= SIMILARITY_THRESHOLD:
        return THEMES[best_idx], sims[best_idx]
    return "Others", sims[best_idx]


# ------------------------------------------------------
# Main Logic
# ------------------------------------------
