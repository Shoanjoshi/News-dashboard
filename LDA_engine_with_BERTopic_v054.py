# ============================================
# 📄 LDA_engine_with_BERTopic_v054.py
# Reconstructed stable engine with:
# - BERTopic clustering + topic summaries
# - Topic embeddings for persistence
# - Embedding-based article→theme assignment (with "Others")
# ============================================

import os
import feedparser
import numpy as np
from openai import OpenAI
from bertopic import BERTopic
from hdbscan import HDBSCAN
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------
# OpenAI client
# --------------------------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --------------------------------------------
# RSS feeds – RESTORED from your config
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
# Themes for embedding-based theme analysis
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

# looser threshold so not everything falls into Others
SIMILARITY_THRESHOLD = 0.30

PROMPT = """You are preparing a factual briefing. Summarize the topic strictly based on the information provided.
Do not infer impact, sentiment, or implications. Avoid subjective language, predictions, or assumptions.
Use neutral, objective tone.

STRICT FORMAT ONLY:
TITLE: <3–5 WORDS, UPPERCASE, factual>
SUMMARY: <2–3 concise factual sentences. No speculation.>"""


# --------------------------------------------
# 1️⃣ RSS article fetcher
# --------------------------------------------
def fetch_articles():
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
                    if len(clean) > 50:
                        docs.append(clean[:1200])
        except Exception as e:
            print(f"⚠ Feed error {feed}: {e}")

    print(f"🔎 RSS articles extracted: {len(docs)}")
    if docs:
        print(f"📌 Sample article: {docs[0][:120]}")
    return docs


# --------------------------------------------
# 2️⃣ GPT topic summarizer
# --------------------------------------------
def gpt_summarize_topic(topic_id, docs_for_topic):
    text = " ".join(docs_for_topic)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": PROMPT + "\n" + text}],
        )
        out = resp.choices[0].message.content or ""
        if "SUMMARY:" in out:
            title_part, summary_part = out.split("SUMMARY:", 1)
            title = title_part.replace("TITLE:", "").strip()
            summary = summary_part.strip()
        else:
            title = f"TOPIC {topic_id}"
            summary = out.strip() or "No summary."
        return {"title": title, "summary": summary}
    except Exception as e:
        print(f"⚠ GPT error on topic {topic_id}: {e}")
        return {
            "title": f"TOPIC {topic_id}",
            "summary": "Summary generation failed.",
        }


# --------------------------------------------
# 3️⃣ BERTopic model runner (in spirit of your v5.5)
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
        min_cluster_size=10,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    vectorizer_model = CountVectorizer(
        stop_words="english",
        max_df=1.0,
        min_df=2,
        ngram_range=(1, 3),
    )

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=True,
        verbose=False,
    )

    topics, probs = topic_model.fit_transform(docs)
    return topic_model, topics, probs


# --------------------------------------------
# 4️⃣ Main entry point used by generate_dashboard.py
# --------------------------------------------
def generate_topic_results():
    docs = fetch_articles()
    if not docs:
        return [], {}, None, {}, {}

    # BERTopic clustering
    topic_model, topics, probs = run_bertopic_analysis(docs)
    topic_info = topic_model.get_topic_info()

    summaries = {}
    topic_embeddings = {}

    # Topic-level summaries + embeddings (used for persistence & topic map)
    for topic_id in topic_info.Topic:
        if topic_id == -1:
            continue
        doc_indices = [i for i, t in enumerate(topics) if t == topic_id]
        if not doc_indices:
            continue
        topic_docs = [docs[i] for i in doc_indices[:5]]
        summaries[topic_id] = gpt_summarize_topic(topic_id, topic_docs)

        # Map topic id -> embedding vector (as list for JSON)
        topic_embeddings[topic_id] = topic_model.topic_embeddings_[topic_id].tolist()

    # ------------------------------------------------
    # 5️⃣ Article→theme assignment via embeddings
    # ------------------------------------------------
    try:
        # Article embeddings from same model BERTopic used
        article_embeddings = topic_model._embedding_model.encode(
            docs, show_progress_bar=False
        )
        # Theme embeddings using same model (same dimensionality)
        theme_embeddings = topic_model._embedding_model.encode(
            THEMES, show_progress_bar=False
        )
    except Exception as e:
        print(f"⚠ Embedding-based theme assignment failed: {e}")
        # Fallback: everything in Others
        theme_metrics = {theme: {"volume": 0, "centrality": 0.0} for theme in THEMES}
        theme_metrics["Others"] = {"volume": len(docs), "centrality": 0.0}
        return docs, summaries, topic_model, topic_embeddings, theme_metrics

    theme_metrics = {theme: {"volume": 0, "centrality": 0.0} for theme in THEMES}
    theme_metrics["Others"] = {"volume": 0, "centrality": 0.0}

    for emb in article_embeddings:
        sims = cosine_similarity([emb], theme_embeddings)[0]
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        if best_score >= SIMILARITY_THRESHOLD:
            assigned_theme = THEMES[best_idx]
        else:
            assigned_theme = "Others"
        theme_metrics[assigned_theme]["volume"] += 1

    print("📊 Theme metrics (volume only):", theme_metrics)

    # NOTE:
    # - summaries is a dict keyed by topic_id (BERTopic topics)
    # - topic_embeddings is a dict keyed by topic_id (used for persistence)
    # - theme_metrics is article-level theme volume (centrality 0.0 for now;
    #   dashboard computes ranks and deltas)
    return docs, summaries, topic_model, topic_embeddings, theme_metrics


# --------------------------------------------
# Local debug
# --------------------------------------------
if __name__ == "__main__":
    d, s, m, e, tm = generate_topic_results()
    print(f"Docs: {len(d)}, topics: {len(s)}")
    print("Themes:", tm)

