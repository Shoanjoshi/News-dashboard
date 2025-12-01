# ============================================
# LDA_engine_with_BERTopic_v054.py
# Stable engine with:
#  - BERTopic clustering + GPT summaries
#  - Article embeddings with MiniLM
#  - Embedding-based article→theme assignment (+Others)
#  - Fallback segmentation when BERTopic collapses
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
from sentence_transformers import SentenceTransformer

# --------------------------------------------
# OpenAI client
# --------------------------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --------------------------------------------
# RSS feeds – restored from your config
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
# Theme definitions
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

SIMILARITY_THRESHOLD = 0.15

PROMPT = """You are preparing a factual briefing. Summarize the topic strictly based on the information provided.
Do not infer impact, sentiment, or implications. Avoid subjective language, predictions, or assumptions.
Use neutral, objective tone.

STRICT FORMAT ONLY:
TITLE: <3–5 WORDS, UPPERCASE, factual>
SUMMARY: <2–3 concise factual sentences. No speculation.>"""


# ========== Utilities ==========

def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


# ========== Fetch RSS Articles ==========

def fetch_articles():
    docs = []
    for feed in RSS_FEEDS:
        try:
            parsed = feedparser.parse(feed)
            for entry in parsed.entries[:20]:
                content = entry.get("summary") or entry.get("description") or entry.get("title") or ""
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


# ========== GPT Topic Summarization ==========

def gpt_summarize_topic(topic_id, docs_for_topic):
    MAX_ARTICLES = 10
    num_docs_for_summary = max(1, min(MAX_ARTICLES, len(docs_for_topic), len(docs_for_topic) // 10 or 1))
    docs_selected = docs_for_topic[:num_docs_for_summary]

    text = "\n\n".join([f"ARTICLE {i+1}:\n{doc}" for i, doc in enumerate(docs_selected)])

    prompt = f"""
You are preparing a factual briefing. Summarize the topic strictly based on the information provided.
Use neutral, objective tone. Do not infer impact, sentiment, predictions, or subjective statements.

STRICT FORMAT ONLY:

TITLE: <3–5 WORDS, UPPERCASE>

KEY POINTS:
- Write ONE factual bullet per article maximum.
- Each bullet must correspond to a separate article.
- NO merging ideas across articles.
- Only core factual statements. No speculation.

Now summarize the following articles:

{text}
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        out = resp.choices[0].message.content or ""

        if "TITLE:" in out:
            parts = out.split("TITLE:", 1)[1].split("\n", 1)
            title = parts[0].strip()
            summary_text = parts[1].strip()
            summary_formatted = "<br>".join([
                f"• {line.strip('-• ').strip()}"
                for line in summary_text.split("\n") if line.strip()
            ])
        else:
            title = f"TOPIC {topic_id}"
            summary_formatted = "Summary format incorrect."

        return {"title": title, "summary": summary_formatted}

    except Exception as e:
        print(f"⚠ GPT error on topic {topic_id}: {e}")
        return {"title": f"TOPIC {topic_id}", "summary": "Summary generation failed."}


# ========== Topic Model ==========

def run_bertopic_analysis(docs):
    umap_model = UMAP(
        n_neighbors=15, 
        n_components=2,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=4,
        min_samples=1,
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


# ========== Main ==========

def generate_topic_results():
    docs = fetch_articles()
    if not docs:
        return [], {}, None, {}, {}

    topic_model, topics, probs = run_bertopic_analysis(docs)
    topic_info = topic_model.get_topic_info()

    summaries = {}
    topic_embeddings = {}

    valid_topic_ids = [t for t in topic_info.Topic if t != -1]

    # ----- Fallback segmentation -----
    MIN_TOPIC_THRESHOLD = 5
    if len(valid_topic_ids) < MIN_TOPIC_THRESHOLD:
        print(f"⚠️ Only {len(valid_topic_ids)} topics detected — activating fallback segmentation.")

        num_fallback_topics = MIN_TOPIC_THRESHOLD
        docs_per_cluster = max(1, len(docs) // num_fallback_topics)

        try:
            embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        except Exception:
            embedding_model = None

        for fallback_id in range(num_fallback_topics):
            start = fallback_id * docs_per_cluster
            end = start + docs_per_cluster
            topic_docs = docs[start:end]
            if not topic_docs:
                continue

            summaries[fallback_id] = gpt_summarize_topic(fallback_id, topic_docs)

            try:
                if embedding_model:
                    emb_slice = embedding_model.encode(topic_docs, show_progress_bar=False)
                    emb_norm = _normalize_rows(np.array(emb_slice))
                    topic_embeddings[fallback_id] = np.mean(emb_norm, axis=0).tolist()
                else:
                    topic_embeddings[fallback_id] = topic_model.topic_embeddings_[0].tolist()
            except Exception:
                topic_embeddings[fallback_id] = topic_model.topic_embeddings_[0].tolist()

        theme_metrics = {theme: {"volume": 0, "centrality": 0.0} for theme in THEMES}
        theme_metrics["Others"] = {"volume": len(docs), "centrality": 0.0}
        return docs, summaries, topic_model, topic_embeddings, theme_metrics

    # ----- Normal Topic Aggregation -----
    for topic_id in valid_topic_ids:
        doc_indices = [i for i, t in enumerate(topics) if t == topic_id]
        if not doc_indices:
            continue
        topic_docs = [docs[i] for i in doc_indices[:5]]
        summaries[topic_id] = gpt_summarize_topic(topic_id, topic_docs)
        topic_embeddings[topic_id] = topic_model.topic_embeddings_[topic_id].tolist()

    # ----- Theme Mapping -----
    try:
        embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        article_embeddings = _normalize_rows(np.array(embedding_model.encode(docs, show_progress_bar=False)))
        theme_embeddings = _normalize_rows(np.array(embedding_model.encode(THEMES, show_progress_bar=False)))
    except Exception:
        theme_metrics = {t: {"volume": 0, "centrality": 0.0} for t in THEMES}
        theme_metrics["Others"] = {"volume": len(docs), "centrality": 0.0}
        return docs, summaries, topic_model, topic_embeddings, theme_metrics

    theme_metrics = {theme: {"volume": 0, "centrality": 0.0} for theme in THEMES}
    theme_metrics["Others"] = {"volume": 0, "centrality": 0.0}

    for emb in article_embeddings:
        sims = cosine_similarity([emb], theme_embeddings)[0]
        best_idx = int(np.argmax(sims))
        assigned_theme = THEMES[best_idx] if sims[best_idx] >= SIMILARITY_THRESHOLD else "Others"
        theme_metrics[assigned_theme]["volume"] += 1

    return docs, summaries, topic_model, topic_embeddings, theme_metrics


# ----- Local Debug -----
if __name__ == "__main__":
    d, s, m, e, tm = generate_topic_results()
    print(f"Docs: {len(d)}, topics: {len(s)}")
    print("Themes:", tm)

