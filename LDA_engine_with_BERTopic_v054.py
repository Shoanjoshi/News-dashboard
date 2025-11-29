# ======================================================
# LDA_engine_with_BERTopic_v054.py  – restored + theme embedding
# ======================================================

import os
import json
import feedparser
import openai
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------
# 🔑 OpenAI API Key
# ------------------------------------------------------
openai.api_key = os.getenv("OPENAI_API_KEY")

# ------------------------------------------------------
# 📡 RSS Feed List – as provided
# ------------------------------------------------------
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

# ------------------------------------------------------
# 🎯 Themes
# ------------------------------------------------------
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

# minimum cosine similarity to assign to a named theme
SIMILARITY_THRESHOLD = 0.5

PROMPT = """You are preparing a factual briefing. Summarize the topic strictly based on the information provided.
Do not infer impact, sentiment, or implications. Avoid subjective language, predictions, or assumptions.
Use neutral, objective tone.

STRICT FORMAT ONLY:
TITLE: <3–5 WORDS, UPPERCASE, factual>
SUMMARY: <2–3 concise factual sentences. No speculation.>"""


# ------------------------------------------------------
# 📰 Fetch RSS Articles
# ------------------------------------------------------
def fetch_rss_articles():
    docs = []
    for feed in RSS_FEEDS:
        try:
            parsed = feedparser.parse(feed)
            for entry in parsed.entries[:20]:
                content = (
                    entry.get("summary")
                    or entry.get("description")
                    or entry.get("title")
                    or ""
                )
                if isinstance(content, str):
                    clean = content.strip()
                    if len(clean) > 50:  # basic filter against junk/very short items
                        docs.append(clean[:1000])
        except Exception as e:
            print(f"⚠ Feed parsing failed: {feed}\nError: {e}")

    print(f"🔎 RSS articles extracted: {len(docs)}")
    if docs:
        print(f"📌 Sample → {docs[0][:120]}")
    return docs


# ------------------------------------------------------
# 🔹 GPT Summary
# ------------------------------------------------------
def summarize_topic(text: str) -> str:
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": PROMPT + "\n" + text}],
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"⚠ GPT summary failed: {e}")
        return "TITLE: UNKNOWN\nSUMMARY: Failed to generate summary."


# ------------------------------------------------------
# 🔍 Embedding Model
# ------------------------------------------------------
def load_embedding_model(topic_model: BERTopic):
    # If BERTopic already has an embedding model, reuse it for themes
    if hasattr(topic_model, "embedding_model") and topic_model.embedding_model:
        return topic_model.embedding_model
    # Fallback – should rarely be needed if BERTopic is configured
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# ------------------------------------------------------
# 🚀 CORE ENGINE — generates topics, summaries, embeddings, theme scores
# ------------------------------------------------------
def generate_topic_results():
    print("🚀 Running topic engine...")
    docs = fetch_rss_articles()

    if not docs:
        print("❌ No articles. Returning empty response.")
        return [], {}, None, {}, {}

    # 1️⃣ Topic modeling with BERTopic
    try:
        topic_model = BERTopic(nr_topics="auto")
        topics, probabilities = topic_model.fit_transform(docs)
    except Exception as e:
        print(f"❌ BERTopic failed: {e}")
        return docs, {}, None, {}, {}

    # 2️⃣ Embeddings for articles
    embedding_model = load_embedding_model(topic_model)
    try:
        embeddings = embedding_model.encode(docs, show_progress_bar=False)
    except Exception as e:
        print(f"❌ Embedding generation failed: {e}")
        return docs, {}, topic_model, {}, {}

    # 3️⃣ GPT-based topic summaries (clustered at BERTopic topic level)
    topic_summaries = {}
    unique_topics = sorted(set(topics))
    for topic_id in unique_topics:
        topic_docs = [docs[i] for i, t in enumerate(topics) if t == topic_id][:3]
        if not topic_docs:
            continue
        combined_text = " ".join(topic_docs).strip()
        summary_response = summarize_topic(combined_text)
        title = summary_response.split("SUMMARY:")[0].replace("TITLE:", "").strip()
        summary_text = summary_response.split("SUMMARY:")[-1].strip()
        topic_summaries[topic_id] = {
            "title": title,
            "summary": summary_text,
            "status": "NEW",  # NEW/PERSISTENT handled later via dashboard persistence
        }

    # 4️⃣ Embedding-based THEMES assignment at ARTICLE level
    print("🧠 Assigning articles to themes using embeddings...")
    try:
        theme_embeddings = embedding_model.encode(THEMES, show_progress_bar=False)
    except Exception as e:
        print(f"⚠ Theme embedding failed, falling back to Others-only. Error: {e}")
        theme_metrics = {theme: {"volume": 0, "centrality": 0.0} for theme in THEMES}
        theme_metrics["Others"] = {"volume": len(docs), "centrality": 0.0}
        print("📊 Theme metrics (fallback):", theme_metrics)
        return docs, topic_summaries, topic_model, embeddings, theme_metrics

    # Initialize metrics
    theme_metrics = {theme: {"volume": 0, "centrality": 0.0} for theme in THEMES}
    theme_metrics["Others"] = {"volume": 0, "centrality": 0.0}

    # For each article embedding, find the closest theme
    for emb in embeddings:
        sims = cosine_similarity([emb], theme_embeddings)[0]
        best_idx = int(np.argmax(sims))
        best_score = sims[best_idx]
        if best_score >= SIMILARITY_THRESHOLD:
            assigned_theme = THEMES[best_idx]
        else:
            assigned_theme = "Others"
        theme_metrics[assigned_theme]["volume"] += 1

    print("📊 Theme metrics (volume only, centrality=0 for now):", theme_metrics)

    # 5️⃣ Return everything for the dashboard
    # embeddings is a numpy array; generate_dashboard iterates over it and saves by index
    return docs, topic_summaries, topic_model, embeddings, theme_metrics


# ------------------------------------------------------
# 🧪 Debug Run
# ------------------------------------------------------
if __name__ == "__main__":
    docs, summaries, model, emb, themes = generate_topic_results()
    print("🧪 DEBUG – theme metrics:")
    print(json.dumps(themes, indent=2))
