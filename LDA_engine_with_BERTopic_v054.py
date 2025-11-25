# ============================================
# 📄 LDA_engine_with_BERTopic_v054.py
# Version 5.4 – Natural topic clustering, expanded RSS, GPT summaries
# ============================================

import os
import feedparser
import re
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
# 2️⃣ RSS SOURCES (earlier feeds + new ones, combined)
# --------------------------------------------
RSS_FEEDS = [
    # --- Existing global business & markets ---
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.reuters.com/reuters/markets",
    "https://feeds.reuters.com/reuters/worldNews",
    "https://feeds.reuters.com/reuters/politicsNews",
    "https://feeds.reuters.com/reuters/environment",
    "https://feeds.reuters.com/reuters/technologyNews",

    "https://www.ft.com/rss/home/us",
    "https://www.ft.com/rss/home/europe",
    "https://www.ft.com/rss/home/asia",

    "https://www.wsj.com/xml/rss/3_7014.xml",   # WSJ - Business
    "https://www.wsj.com/xml/rss/3_7085.xml",   # WSJ - Markets

    "https://feeds.marketwatch.com/marketwatch/topstories/",
    "https://feeds.marketwatch.com/marketwatch/marketpulse/",

    "http://feeds.bbci.co.uk/news/world/rss.xml",
    "http://feeds.bbci.co.uk/news/business/rss.xml",

    "http://rss.cnn.com/rss/edition_world.rss",
    "http://rss.cnn.com/rss/edition_business.rss",

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

    # --- New feeds you provided (market anomalies, leverage, regs etc.) ---

    # Mainstream financial news with market anomalies occasionally highlighted
    "https://www.ft.com/rss/home",                           # FT home
    "https://www.marketwatch.com/rss/topstories",            # MarketWatch – Top Stories
    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",         # WSJ – Markets main

    # High-frequency market signals and speculative behavior
    "https://www.investing.com/rss/news_25.rss",             # Crypto
    "https://www.investing.com/rss/news_1.rss",              # Stock Market
    "https://www.investing.com/rss/news_285.rss",            # Market Overview

    # Macro stats & consumer debt (signs of leverage)
    "https://www.federalreserve.gov/feeds/data.xml",         # Fed – Data
    "https://www.federalreserve.gov/feeds/press_all.xml",    # Fed – All Press

    # Extreme investor behavior / bubble / HF commentary
    "https://markets.businessinsider.com/rss",               # BI – Markets

    # Structured products, derivatives, and alternative risk commentary
    "https://www.risk.net/feeds/rss",                        # Risk.net

    # Commentary on nontraditional assets
    "https://www.forbes.com/finance/feed",                   # Forbes Finance
    "https://feeds.feedburner.com/alternativeinvestmentnews",# Alt investments

    # Regulatory and supervisory updates (systemic stress signals)
    "https://www.eba.europa.eu/eba-news-rss",                # EBA
    "https://www.bis.org/rss/press_rss.xml",                 # BIS
    "https://www.imf.org/external/np/exr/feeds/rss.aspx?type=imfnews", # IMF news
]

# --------------------------------------------
# 3️⃣ Fetch & Clean Articles
# --------------------------------------------
def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    text = re.sub(r"<[^>]+>", "", text)
    return text

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
    if len(articles) < 15:
        print("⚠️ Not enough articles for stable topics.")
        return []
    return articles

# --------------------------------------------
# 4️⃣ GPT Summarization (original prompt unchanged)
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
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=350,
        )
        content = resp.choices[0].message.content.strip()
        title, summary = None, None
        for line in content.split("\n"):
            lower = line.strip().lower()
            if lower.startswith("title:"):
                title = line.split(":", 1)[1].strip()
            elif lower.startswith("summary:"):
                summary = line.split(":", 1)[1].strip()

        if not title:
            title = f"Topic {topic_id}"
        if not summary:
            summary = ", ".join(words[:5]) + " (fallback)"

        return {"title": title, "summary": summary}

    except Exception as e:
        print(f"⚠️ GPT error (fallback): {e}")
        return {"title": f"Topic {topic_id}",
                "summary": ", ".join(words[:5]) + " (fallback)"}

# --------------------------------------------
# 5️⃣ BERTopic Model
# --------------------------------------------
def run_bertopic_analysis(docs):
    umap_model = UMAP(
        n_neighbors=15,
        n_components=2,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=8,
        min_samples=1,
    )

    vectorizer_model = CountVectorizer(stop_words="english")

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        nr_topics=None,                # natural topic discovery
        top_n_words=15,
        vectorizer_model=vectorizer_model,
        verbose=True,
    )

    topics, probs = topic_model.fit_transform(docs)
    return topic_model, topics, probs

# --------------------------------------------
# 6️⃣ Main Runner – returns docs, summaries, model, embeddings
# --------------------------------------------
def generate_topic_results():
    docs = fetch_articles()
    if not docs:
        return [], {}, None, None

    topic_model, topics, probs = run_bertopic_analysis(docs)
    topic_info = topic_model.get_topic_info()

    summaries = {}
    embeddings = {}   # topic_id -> embedding list

    for topic_id in topic_info.Topic:
        if topic_id == -1:
            continue
        words = [w[0] for w in topic_model.get_topic(topic_id)]
        feat_docs = [docs[i] for i, t in enumerate(topics) if t == topic_id]
        if not feat_docs:
            continue

        summaries[topic_id] = summarize_topic_gpt(topic_id, words, feat_docs)
        # store as list so it can be JSON serialized later
        embeddings[topic_id] = topic_model.topic_embeddings_[topic_id].tolist()

    return docs, summaries, topic_model, embeddings

# --------------------------------------------
# 7️⃣ Local test
# --------------------------------------------
if __name__ == "__main__":
    docs, summaries, model, emb = generate_topic_results()
    print("📊 Topic summaries:")
    for k, v in summaries.items():
        print(k, "→", v["title"])
