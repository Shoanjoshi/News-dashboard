# ============================================
# 📄 LDA_engine_with_BERTopic_v054.py
# Version 5.5 – Natural topics + persistence (stable)
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
# 1️⃣ OpenAI Client
# --------------------------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Persistence tracking settings
PREVIOUS_TOPICS_JSON = os.path.join("dashboard", "yesterday_topics.json")
SIMILARITY_THRESHOLD = 0.75  # Cosine similarity threshold for persistence

# --------------------------------------------
# 2️⃣ RSS Sources
# (Existing + new — unchanged from your last working version)
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
# 3️⃣ Fetch & Clean
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
    return articles if len(articles) >= 15 else []

# --------------------------------------------
# 4️⃣ GPT Summarization (factual style)
# --------------------------------------------
def summarize_topic_gpt(topic_id, words, docs):
    snippet_text = "\n".join(f"- {d[:200]}..." for d in docs[:3])

    prompt = (
        "You are preparing a factual briefing.\n"
        "Write an objective summary strictly based on the provided content.\n"
        "Do not infer impact, sentiment, or speculation.\n\n"
        "FORMAT:\n"
        "TITLE: <3–5 WORDS, UPPERCASE, factual>\n"
        "SUMMARY: <2–3 factual sentences based only on available information>\n"
        f"Topic ID: {topic_id}\n"
        f"Key Terms: {', '.join(words[:10])}\n"
        f"Example Snippets:\n{snippet_text}"
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=300,
        )
        content = resp.choices[0].message.content.strip()

        title, summary = None, None
        for line in content.split("\n"):
            lower = line.lower().strip()
            if lower.startswith("title:"):
                title = line.split(":", 1)[1].strip()
            elif lower.startswith("summary:"):
                summary = line.split(":", 1)[1].strip()

        return {
            "title": title or f"Topic {topic_id}",
            "summary": summary or ", ".join(words[:5]) + " (fallback)"
        }

    except Exception as e:
        print(f"⚠️ GPT error: {e}")
        return {"title": f"Topic {topic_id}", "summary": ", ".join(words[:5]) + " (fallback)"}

# --------------------------------------------
# 5️⃣ RESTORED – BERTopic model runner
# --------------------------------------------
def run_bertopic_analysis(docs):
    umap_model = UMAP(n_neighbors=15, n_components=2, min_dist=0.1, metric="cosine", random_state=42)
    hdbscan_model = HDBSCAN(min_cluster_size=8, min_samples=1)
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
# 6️⃣ Persistence: Label New / Persistent
# --------------------------------------------
def _cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def _load_previous_embeddings():
    if not os.path.exists(PREVIOUS_TOPICS_JSON):
        print("🟡 No previous topics found – treating all as new.")
        return {}
    try:
        with open(PREVIOUS_TOPICS_JSON, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {int(k): np.array(v["embedding"]) for k, v in raw.items()}
    except Exception as e:
        print(f"⚠️ Error loading previous topics: {e}")
        return {}

def label_persistence(current_embeddings):
    prev_embs = _load_previous_embeddings()
    labels = {}
    for tid, emb in current_embeddings.items():
        best_sim = max((_cosine_similarity(emb, p) for p in prev_embs.values()), default=0.0)
        labels[tid] = "PERSISTENT" if best_sim >= SIMILARITY_THRESHOLD else "NEW"
    return labels

# --------------------------------------------
# 7️⃣ Main – Returns 5 values (required)
# --------------------------------------------
def generate_topic_results():
    docs = fetch_articles()
    if not docs:
        return [], {}, None, None, {}  # Return empty placeholders

    topic_model, topics, probs = run_bertopic_analysis(docs)
    topic_info = topic_model.get_topic_info()

    summaries, embeddings = {}, {}

    for topic_id in topic_info.Topic:
        if topic_id == -1:
            continue
        feat_docs = [docs[i] for i, t in enumerate(topics) if t == topic_id]
        if not feat_docs:
            continue

        words = [w[0] for w in topic_model.get_topic(topic_id)]
        summaries[topic_id] = summarize_topic_gpt(topic_id, words, feat_docs)
        embeddings[topic_id] = topic_model.topic_embeddings_[topic_id].tolist()

    persistence = label_persistence(embeddings)

    for tid in summaries:
        summaries[tid]["status"] = persistence.get(tid, "NEW")

    return docs, summaries, topic_model, embeddings, {}  # {} placeholder for theme_scores

# --------------------------------------------
# 8️⃣ Local Test
# --------------------------------------------
if __name__ == "__main__":
    docs, summaries, model, emb, _ = generate_topic_results()
    for k, v in summaries.items():
        print(f"{k} – {v.get('title')} [{v.get('status')}]")
