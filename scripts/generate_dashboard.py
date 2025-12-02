# ============================================
# LDA_engine_with_BERTopic_v054.py
# Stable engine with:
#  - BERTopic clustering using fixed clusters via KMeans
#  - GPT summaries with holistic topic summaries
#  - Expanded RSS feeds (NYT, WaPo, Cybersecurity sources)
#  - Multi-theme assignment for topicality & centrality
#  - Theme article lists (articles_raw) exposed for heatmap
# ============================================

import os
import feedparser
import numpy as np
from collections import Counter

from openai import OpenAI
from bertopic import BERTopic
from sklearn.cluster import KMeans
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# --------------------------------------------
# OpenAI client
# --------------------------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --------------------------------------------
# Expanded RSS feeds
# --------------------------------------------
RSS_FEEDS = [
    # Existing major feeds
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.reuters.com/reuters/markets",
    "https://www.ft.com/rss/home/us",
    "https://www.wsj.com/xml/rss/3_7014.xml",
    "https://www.wsj.com/xml/rss/3_7085.xml",
    "https://feeds.marketwatch.com/marketwatch/topstories/",
    "https://feeds.marketwatch.com/marketwatch/marketpulse/",
    "http://feeds.bbci.co.uk/news/business/rss.xml",
    "http://rss.cnn.com/rss/edition_business.rss",

    # International
    "https://www.ft.com/rss/home/europe",
    "https://www.ft.com/rss/home/asia",
    "https://asia.nikkei.com/rss/feed",
    "https://www.scmp.com/rss/91/feed",

    # Technology
    "https://feeds.reuters.com/reuters/technologyNews",
    "https://feeds.feedburner.com/TechCrunch/",

    # Macro + Risk + Regulation
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

    # NEW FEED ADDITIONS FOR VARIETY
    "https://rss.nytimes.com/services/xml/rss/nyt/Economy.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",
    "https://feeds.washingtonpost.com/rss/business",
    "https://feeds.washingtonpost.com/rss/business/economy",

    # Cybersecurity feeds
    "https://krebsonsecurity.com/feed/",
    "https://www.bleepingcomputer.com/feed/",
    "https://www.darkreading.com/rss.xml",
    "https://www.scmagazine.com/section/feed",

    # Technology + risk
    "https://feeds.arstechnica.com/arstechnica/technology-lab",
]

# --------------------------------------------
# Themes
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

THEME_DESCRIPTIONS = {
    "Recessionary pressures": "Economic slowdown, declining demand, unemployment risk, or business contraction.",
    "Inflation": "Persistent price increases and monetary policy response impacting costs and purchasing power.",
    "Private credit": "Non-bank lending, private debt fund activity, liquidity constraints, and leveraged finance risk.",
    "AI": "Artificial intelligence development, enterprise adoption, automation, and regulatory concerns.",
    "Cyber attacks": "Cybersecurity breaches, systemic risks related to data or technology vulnerabilities.",
    "Commercial real estate": "Office, retail, industrial, and hospitality trends with refinancing risk.",
    "Consumer debt": "Household financial pressure, delinquencies, affordability trends.",
    "Bank lending and credit risk": "Credit exposure, regulatory pressure, default risk in banking portfolios.",
    "Others": "Articles not clearly tied to systemic themes.",
}

SIMILARITY_THRESHOLD = 0.20

# --------------------------------------------
# Normalize rows
# --------------------------------------------
def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms

# --------------------------------------------
# Fetch articles
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
    return docs

# --------------------------------------------
# GPT summarizer
# --------------------------------------------
def gpt_summarize_topic(topic_id, docs_for_topic):
    MAX_SUMMARY_DOCS = 8
    docs_selected = docs_for_topic[:MAX_SUMMARY_DOCS]

    text = "\n\n".join([f"ARTICLE:\n{d}" for d in docs_selected])

    prompt = f"""
You are preparing an objective briefing based ONLY on the provided content.
Summarize the central theme of the topic. No speculation.

TITLE: <3–5 WORDS, UPPERCASE>
SUMMARY: <2–4 concise sentences>

{text}
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        out = resp.choices[0].message.content or ""

        if "TITLE:" in out and "SUMMARY:" in out:
            title = out.split("TITLE:", 1)[1].split("SUMMARY:", 1)[0].strip()
            summary = out.split("SUMMARY:", 1)[1].strip().replace("\n", " ")
        else:
            title, summary = f"TOPIC {topic_id}", "Summary unavailable."

        return {"title": title, "summary": summary}

    except Exception as e:
        print(f"⚠ GPT error for topic {topic_id}: {e}")
        return {"title": f"TOPIC {topic_id}", "summary": "Summary failed."}

# --------------------------------------------
# BERTopic model
# --------------------------------------------
def run_bertopic_analysis(docs):
    umap_model = UMAP(
        n_neighbors=30, n_components=2, min_dist=0.0,
        metric="cosine", random_state=42
    )

    kmeans_model = KMeans(
        n_clusters=15, random_state=42, n_init="auto"
    )

    vectorizer_model = CountVectorizer(
        stop_words="english", max_df=1.0, min_df=2, ngram_range=(1, 3)
    )

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=kmeans_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=True,
        verbose=False,
    )

    topics, probs = topic_model.fit_transform(docs)
    return topic_model, topics, probs

# --------------------------------------------
# Main engine
# --------------------------------------------
def generate_topic_results():
    docs = fetch_articles()
    if not docs:
        return [], {}, None, {}, {}

    topic_model, topics, probs = run_bertopic_analysis(docs)
    topic_info = topic_model.get_topic_info()

    valid_topic_ids = [t for t in topic_info.Topic if t != -1]

    summaries, topic_embeddings = {}, {}

    # Summaries + embeddings
    for topic_id in valid_topic_ids:
        doc_idx = [i for i, t in enumerate(topics) if t == topic_id]
        if not doc_idx:
            continue

        topic_docs = [docs[i] for i in doc_idx[:5]]
        out = gpt_summarize_topic(topic_id, topic_docs)
        out["article_count"] = len(doc_idx)
        summaries[topic_id] = out

        topic_embeddings[topic_id] = topic_model.topic_embeddings_[topic_id].tolist()

    # --------------------------------------------
    # Theme assignment via embeddings
    # --------------------------------------------
    try:
        embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        # Articles
        article_embeddings = embed_model.encode(docs, show_progress_bar=False)
        article_embeddings = _normalize_rows(np.array(article_embeddings))

        # Themes
        theme_texts = [
            f"{t}. {THEME_DESCRIPTIONS.get(t, '')}" for t in THEMES
        ]
        theme_embeddings = embed_model.encode(theme_texts, show_progress_bar=False)
        theme_embeddings = _normalize_rows(np.array(theme_embeddings))

    except Exception as e:
        print("⚠ Embedding error:", e)
        # fallback
        return docs, summaries, topic_model, topic_embeddings, {
            t: {"volume": 0, "centrality": 0.0} for t in THEMES
        }

    # Initialize theme metrics
    theme_metrics = {
        t: {"volume": 0, "centrality": 0.0, "articles_raw": []}
        for t in THEMES
    }
    theme_metrics["Others"] = {"volume": 0, "centrality": 0.0, "articles_raw": []}

    # Assign articles to all themes over threshold
    for i, emb in enumerate(article_embeddings):
        sims = cosine_similarity([emb], theme_embeddings)[0]

        assigned = [
            THEMES[idx] for idx, s in enumerate(sims)
            if s >= SIMILARITY_THRESHOLD
        ]

        if not assigned:
            assigned = ["Others"]

        for theme in assigned:
            theme_metrics[theme]["volume"] += 1
            theme_metrics[theme]["articles_raw"].append(i)

    # --------------------------------------------
    # Centrality = overlap of article sets
    # --------------------------------------------
    for theme in THEMES:
        total_overlap = 0
        A = set(theme_metrics[theme]["articles_raw"])

        for other in THEMES:
            if other == theme:
                continue
            B = set(theme_metrics[other]["articles_raw"])
            total_overlap += len(A & B)

        theme_metrics[theme]["centrality_raw"] = total_overlap

    # Normalize
    max_ov = max(theme_metrics[t]["centrality_raw"] for t in THEMES) or 1
    for t in THEMES:
        theme_metrics[t]["centrality"] = theme_metrics[t]["centrality_raw"] / max_ov

    theme_metrics["Others"]["centrality"] = 0.0

    # Cleanup helper key
    for t in theme_metrics:
        theme_metrics[t].pop("centrality_raw", None)

    print("📊 Theme metrics:", theme_metrics)
    return docs, summaries, topic_model, topic_embeddings, theme_metrics


# --------------------------------------------
# Debug
# --------------------------------------------
if __name__ == "__main__":
    d, s, m, e, tm = generate_topic_results()
    print("Docs:", len(d))
    print("Themes:", tm)
