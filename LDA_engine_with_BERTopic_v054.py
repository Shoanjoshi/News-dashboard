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

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

RSS_FEEDS = [
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.reuters.com/reuters/markets",
    "https://www.ft.com/rss/home/us",
    "https://www.wsj.com/xml/rss/3_7014.xml",
    "https://www.wsj.com/xml/rss/3_7085.xml",
    "https://feeds.marketwatch.com/marketwatch/topstories/",
    "https://feeds.marketwatch.com/marketwatch/marketpulse/",
    "http://feeds.bbci.co.uk/news/business/rss.xml",
    "http://rss.cnn.com/rss/edition_business.rss",
    "https://www.ft.com/rss/home/europe",
    "https://www.ft.com/rss/home/asia",
    "https://asia.nikkei.com/rss/feed",
    "https://www.scmp.com/rss/91/feed",
    "https://feeds.reuters.com/reuters/technologyNews",
    "https://feeds.feedburner.com/TechCrunch/",
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
    "https://rss.nytimes.com/services/xml/rss/nyt/Economy.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",
    "https://feeds.washingtonpost.com/rss/business",
    "https://feeds.washingtonpost.com/rss/business/economy",
    "https://krebsonsecurity.com/feed/",
    "https://www.bleepingcomputer.com/feed/",
    "https://www.darkreading.com/rss.xml",
    "https://www.scmagazine.com/section/feed",
    "https://feeds.arstechnica.com/arstechnica/technology-lab",
]

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
    "Recessionary pressures": "Economic slowdown, declining demand.",
    "Inflation": "Price increases and monetary policy.",
    "Private credit": "Non-bank lending and liquidity risk.",
    "AI": "Artificial intelligence and automation trends.",
    "Cyber attacks": "Security breaches and vulnerabilities.",
    "Commercial real estate": "Property market stress and refinancing.",
    "Consumer debt": "Household leverage and affordability issues.",
    "Bank lending and credit risk": "Defaults and regulatory pressure.",
    "Others": "Articles not matching systemic themes.",
}

SIMILARITY_THRESHOLD = 0.20


def _normalize_rows(mat):
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return mat / norms


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
                if isinstance(content, str) and len(content.strip()) > 50:
                    docs.append(content.strip()[:1200])
        except Exception as e:
            print(f"Feed error {feed}: {e}")

    print("Fetched articles:", len(docs))
    return docs


def gpt_summarize_topic(topic_id, docs_for_topic):
    text = "\n\n".join(docs_for_topic[:8])

    prompt = f"""
TITLE: <3–5 WORDS>
SUMMARY: <2–4 sentences>

Content:
{text}
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        out = resp.choices[0].message.content or ""
        if "TITLE:" in out and "SUMMARY:" in out:
            t = out.split("TITLE:", 1)[1].split("SUMMARY:", 1)
            return {"title": t[0].strip(), "summary": t[1].strip()}
    except:
        pass

    return {"title": f"TOPIC {topic_id}", "summary": "Summary unavailable."}


def run_bertopic_analysis(docs):
    umap_model = UMAP(n_neighbors=30, n_components=2, min_dist=0.0, metric="cosine")
    kmeans_model = KMeans(n_clusters=15, random_state=42, n_init="auto")
    vectorizer_model = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 3))

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=kmeans_model,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=True,
    )
    topics, probs = topic_model.fit_transform(docs)
    return topic_model, topics


def generate_topic_results():
    docs = fetch_articles()
    if not docs:
        return [], {}, None, {}, {}

    topic_model, topics = run_bertopic_analysis(docs)
    topic_info = topic_model.get_topic_info()

    valid_topic_ids = [t for t in topic_info.Topic if t != -1]

    summaries = {}
    topic_embeddings = {}

    for topic_id in valid_topic_ids:
        doc_ids = [i for i, t in enumerate(topics) if t == topic_id]
        topic_docs = [docs[i] for i in doc_ids[:5]]
        summaries[topic_id] = gpt_summarize_topic(topic_id, topic_docs)
        summaries[topic_id]["article_count"] = len(doc_ids)
        topic_embeddings[topic_id] = topic_model.topic_embeddings_[topic_id].tolist()

    # --------------------------------------------------------------------
    # THEME ASSIGNMENT (heatmap fix: keep article lists)
    # --------------------------------------------------------------------
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    art_emb = _normalize_rows(model.encode(docs, show_progress_bar=False))

    theme_texts = [f"{t}. {THEME_DESCRIPTIONS[t]}" for t in THEMES]
    theme_emb = _normalize_rows(model.encode(theme_texts, show_progress_bar=False))

    theme_metrics = {
        t: {"volume": 0, "centrality": 0.0, "articles": set()}
        for t in THEMES
    }
    theme_metrics["Others"] = {"volume": 0, "centrality": 0.0, "articles": set()}

    for i, emb in enumerate(art_emb):
        sims = cosine_similarity([emb], theme_emb)[0]
        assigned = [
            THEMES[idx]
            for idx, score in enumerate(sims)
            if score >= SIMILARITY_THRESHOLD
        ]
        if not assigned:
            assigned = ["Others"]

        for theme in assigned:
            theme_metrics[theme]["volume"] += 1
            theme_metrics[theme]["articles"].add(i)

    # CENTRALITY
    for t in THEMES:
        overlaps = 0
        Ta = theme_metrics[t]["articles"]
        for other in THEMES:
            if other != t:
                overlaps += len(Ta.intersection(theme_metrics[other]["articles"]))
        theme_metrics[t]["centrality_raw"] = overlaps

    max_c = max(theme_metrics[t].get("centrality_raw", 0) for t in THEMES) or 1
    for t in THEMES:
        theme_metrics[t]["centrality"] = theme_metrics[t]["centrality_raw"] / max_c

    theme_metrics["Others"]["centrality"] = 0.0

    # CLEANUP but keep article IDs for heatmap
    for t in theme_metrics:
        theme_metrics[t].pop("centrality_raw", None)
        theme_metrics[t]["articles_raw"] = list(theme_metrics[t]["articles"])

    return docs, summaries, topic_model, topic_embeddings, theme_metrics


if __name__ == "__main__":
    d, s, m, e, tm = generate_topic_results()
    print("Docs:", len(d))
    print("Themes:", tm)
