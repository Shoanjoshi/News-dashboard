
# ============================================
# LDA_engine_with_BERTopic_v_fixed_clusters.py
# Stable engine with:
#  - BERTopic clustering using fixed clusters via KMeans
#  - GPT summaries with holistic topic summaries
#  - Expanded RSS feeds (NYT, WaPo, Cybersecurity sources)
#  - Multi-theme assignment for topicality & centrality
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

    # Major US newspapers
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
# Theme definitions
# --------------------------------------------

# Core theme labels (no "Others" here)
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
    "Commercial real estate": "Trends in office, retail, industrial, and hospitality real estate with refinancing risk.",
    "Consumer debt": "Household financial pressure, debt levels, delinquencies, affordability shifts.",
    "Bank lending and credit risk": "Credit exposure in banking portfolios, regulatory pressure, and default risks.",
    "Others": "Articles not directly linked to financial systemic themes.",
}

SIMILARITY_THRESHOLD = 0.20  # for assigning themes


# --------------------------------------------
# Helper normalization
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
    if docs:
        print(f"📌 Sample article: {docs[0][:120]}")
    return docs


# --------------------------------------------
# GPT topic summarizer (holistic)
# --------------------------------------------
def gpt_summarize_topic(topic_id, docs_for_topic):
    """
    Improved topic summarization using representative docs with holistic summary.
    Ensures summaries match topic map keywords and removes bullet formatting.
    """
    MAX_SUMMARY_DOCS = 8  # use more docs for better representation
    docs_selected = docs_for_topic[:MAX_SUMMARY_DOCS]

    text = "\n\n".join([f"ARTICLE:\n{doc}" for doc in docs_selected])

    prompt = f"""
You are preparing an objective briefing based ONLY on the content provided.
Summarize the central theme of the topic without referencing individual articles.
Avoid subjective tone, speculation, or predictions. Stick to facts.

STRICT FORMAT:

TITLE: <3–5 WORDS, UPPERCASE, factual>
SUMMARY: <2–4 concise sentences that capture the main theme. No bullet points. No article separation.>

Content to analyze:
{text}
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        out = resp.choices[0].message.content or ""

        if "TITLE:" in out and "SUMMARY:" in out:
            parts = out.split("TITLE:", 1)[1].split("SUMMARY:", 1)
            title = parts[0].strip()
            summary_text = parts[1].strip()
            summary_formatted = summary_text.replace("\n", " ")
        else:
            title = f"TOPIC {topic_id}"
            summary_formatted = "Summary format incorrect."

        return {"title": title, "summary": summary_formatted}

    except Exception as e:
        print(f"⚠ GPT error on topic {topic_id}: {e}")
        return {
            "title": f"TOPIC {topic_id}",
            "summary": "Summary generation failed.",
        }


# --------------------------------------------
# BERTopic model (fixed clusters)
# --------------------------------------------
def run_bertopic_analysis(docs):
    umap_model = UMAP(
        n_neighbors=30,
        n_components=2,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )

    kmeans_model = KMeans(
        n_clusters=15,
        random_state=42,
        n_init="auto",
    )

    vectorizer_model = CountVectorizer(
        stop_words="english",
        max_df=1.0,
        min_df=2,
        ngram_range=(1, 3),
    )

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=kmeans_model,  # keep as-is per your working setup
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

    if len(valid_topic_ids) < 5:
        print(f"⚠ Only {len(valid_topic_ids)} topics detected — but continuing with KMeans output.")

    for topic_id in valid_topic_ids:
        doc_indices = [i for i, t in enumerate(topics) if t == topic_id]
        if not doc_indices:
            continue
        topic_docs = [docs[i] for i in doc_indices[:5]]
        summaries[topic_id] = gpt_summarize_topic(topic_id, topic_docs)
        topic_embeddings[topic_id] = topic_model.topic_embeddings_[topic_id].tolist()

    # ------------------------------------------------
    # Article→theme assignment via SentenceTransformer embeddings
    # ------------------------------------------------
    try:
        embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        # Encode articles
        article_embeddings = embedding_model.encode(
            docs, show_progress_bar=False
        )
        article_embeddings = _normalize_rows(np.array(article_embeddings))

        # Encode themes using title + description (no Others here)
        theme_texts = [
            f"{theme}. {THEME_DESCRIPTIONS.get(theme, '')}"
            for theme in THEMES
        ]
        theme_embeddings = embedding_model.encode(
            theme_texts, show_progress_bar=False
        )
        theme_embeddings = _normalize_rows(np.array(theme_embeddings))

    except Exception as e:
        print(f"⚠ Embedding-based theme assignment failed: {e}")
        theme_metrics = {t: {"volume": 0, "centrality": 0.0} for t in THEMES}
        theme_metrics["Others"] = {"volume": len(docs), "centrality": 0.0}
        return docs, summaries, topic_model, topic_embeddings, theme_metrics

    # Initialize theme metrics (include Others)
    theme_metrics = {
        t: {"volume": 0, "centrality": 0.0, "articles": set()}
        for t in THEMES
    }
    theme_metrics["Others"] = {"volume": 0, "centrality": 0.0, "articles": set()}

    # Multi-theme assignment: each article can map to 0..N themes
    for i, emb in enumerate(article_embeddings):
        sims = cosine_similarity([emb], theme_embeddings)[0]

        assigned_themes = [
            THEMES[idx]
            for idx, score in enumerate(sims)
            if score >= SIMILARITY_THRESHOLD
        ]

        if not assigned_themes:
            assigned_themes = ["Others"]

        for theme in assigned_themes:
            theme_metrics[theme]["volume"] += 1
            theme_metrics[theme]["articles"].add(i)

    # ---- Calculate centrality (overlap of articles across themes) ----
    for theme in THEMES:  # exclude Others from centrality computation
        overlaps = 0
        theme_articles = theme_metrics[theme]["articles"]

        for other_theme in THEMES:
            if other_theme != theme:
                other_articles = theme_metrics[other_theme]["articles"]
                overlaps += len(theme_articles.intersection(other_articles))

        theme_metrics[theme]["centrality_raw"] = overlaps

    # Normalize centrality 0–1
    max_overlap = max(
        [theme_metrics[t].get("centrality_raw", 0) for t in THEMES]
    ) or 1
    for t in THEMES:
        theme_metrics[t]["centrality"] = theme_metrics[t]["centrality_raw"] / max_overlap

    # Others centrality = 0 by design
    theme_metrics["Others"]["centrality"] = 0.0

    # Cleanup helper fields
    for t in list(theme_metrics.keys()):
        theme_metrics[t].pop("articles", None)
        theme_metrics[t].pop("centrality_raw", None)

    print(f"📊 Theme metrics:", theme_metrics)
    print("\n=== DEBUG: Topic Distribution ===\n", Counter(topics))

    return docs, summaries, topic_model, topic_embeddings, theme_metrics


# --------------------------------------------
# Local debug
# --------------------------------------------
if __name__ == "__main__":
    d, s, m, e, tm = generate_topic_results()
    print(f"Docs: {len(d)}, topics: {len(s)}")
    print("Themes:", tm)
