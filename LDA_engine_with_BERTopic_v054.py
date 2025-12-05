import os
import feedparser
import numpy as np
from collections import Counter
from textwrap import wrap

from openai import OpenAI
from bertopic import BERTopic
from sklearn.cluster import KMeans
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import plotly.graph_objects as go

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ============================================================
# CONFIG
# ============================================================
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
    "https://www.bls.gov/feed/news-release.htm?view=all&format=rss",
    "https://www.bea.gov/rss.xml",
    "https://www.cbo.gov/publications/all/rss.xml",
    "https://fredblog.stlouisfed.org/feed/",
    "https://libertystreeteconomics.newyorkfed.org/feed/",
    "https://pitchbook.com/news/feed",
    "https://www.preqin.com/insights/rss",
    "https://www.privatedebtinvestor.com/feed/",
    "https://www.directlendingdeals.com/rss",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://www.theblock.co/rss",
    "https://blog.chainalysis.com/feed/",
    "https://www.trmlabs.com/blog?format=rss",
    "https://cryptoslate.com/feed/",
    "https://cointelegraph.com/rss",
    "https://www.circle.com/blog/rss.xml",
    "https://tether.to/en/feed/",
    "https://forum.makerdao.com/latest.rss"
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
    "Digital assets"
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
    "Digital assets": "Crypto markets, stablecoins, tokenization, blockchain infrastructure, regulatory developments, and risks related to systemic spillovers into traditional finance.",
    "Others": "Articles not matching systemic themes.",
}

SIMILARITY_THRESHOLD = 0.20

# ============================================================
# HELPERS
# ============================================================
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


def get_representative_doc_ids(doc_ids, doc_embeddings, top_k=8):
    """
    Return the indices of the most representative documents for a topic.

    We compute the centroid of the topic's document embeddings and
    then select the top_k documents by cosine similarity to this centroid.
    """
    if not doc_ids:
        return []
    if len(doc_ids) <= top_k:
        return doc_ids

    emb = doc_embeddings[doc_ids]  # (n_docs_in_topic, dim)
    centroid = np.mean(emb, axis=0, keepdims=True)
    sims = cosine_similarity(emb, centroid).ravel()
    ranked = np.argsort(-sims)
    return [doc_ids[i] for i in ranked[:top_k]]


def gpt_summarize_topic(topic_id, docs_for_topic):
    """
    Structured, sharper topic summary with:
      - TITLE
      - OVERVIEW (1–2 sentences)
      - KEY EXAMPLES (2–4 bullets)
    """
    # Use all provided docs_for_topic (already representative)
    articles_block = "\n\n".join(
        [f"ARTICLE {i+1}:\n{doc}" for i, doc in enumerate(docs_for_topic)]
    )

    prompt = f"""
You are summarizing a news topic formed by clustering multiple related articles.

Write a structured, factual, concise summary in this exact layout:

TITLE: <3–5 word topic label>

OVERVIEW:
1–2 sentences summarizing the main common theme across these articles.
Be concrete and specific. Avoid vague macro language and grand conclusions.

KEY EXAMPLES:
- Short, distinct example 1 drawn from one article
- Short, distinct example 2 drawn from another article
- Short, distinct example 3 (optional)
- Short, distinct example 4 (optional)

Rules:
- Use only information that appears in the articles.
- Do not invent entities, events, or numbers.
- Do not mention specific publishers or dates.
- Do not explain your reasoning or mention this prompt.

ARTICLES:
{articles_block}
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        out = resp.choices[0].message.content or ""

        # Robust parsing: take first line after "TITLE:" as the title,
        # and everything after that as the summary body.
        if "TITLE:" in out:
            _, after_title = out.split("TITLE:", 1)
            after_title = after_title.strip()
            lines = after_title.splitlines()

            if lines:
                title_line = lines[0].strip()
                summary_body = "\n".join(lines[1:]).strip()
            else:
                title_line = f"TOPIC {topic_id}"
                summary_body = out.strip()

            return {
                "title": title_line,
                "summary": summary_body if summary_body else "Summary unavailable.",
            }

    except Exception as e:
        print(f"GPT error for topic {topic_id}: {e}")

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

# ============================================================
# BUILD IMPROVED TOPIC MAP
# ============================================================
def build_topic_map(topic_embeddings, summaries):
    topic_ids = sorted(topic_embeddings.keys())
    xs = [topic_embeddings[i][0] for i in topic_ids]
    ys = [topic_embeddings[i][1] for i in topic_ids]

    volumes = []
    titles = {}
    for item in summaries.values():
        tid = item.get("topic_id") or list(summaries.keys())[list(summaries.values()).index(item)]
        titles[tid] = item["title"]
        volumes.append(item["article_count"])

    size_scale = np.interp(volumes, (min(volumes), max(volumes)), (25, 70))
    labels = {tid: "<br>".join(wrap(titles[tid], width=18)) for tid in topic_ids}

    label_offsets = {}
    for i, tid in enumerate(topic_ids):
        x, y = xs[i], ys[i]
        offset_x = 0.02 * (i % 3 - 1)
        offset_y = 0.03 * ((i // 3) % 3 - 1)
        label_offsets[tid] = (x + offset_x, y + offset_y)

    scatter = go.Scatter(
        x=xs,
        y=ys,
        mode="markers",
        marker=dict(
            size=size_scale,
            color="rgba(60,120,180,0.25)",
            line=dict(color="rgba(60,120,180,0.9)", width=2),
        ),
        text=[titles[i] for i in topic_ids],
        hovertemplate="<b>%{text}</b><extra></extra>",
    )

    label_scatter = go.Scatter(
        x=[label_offsets[tid][0] for tid in topic_ids],
        y=[label_offsets[tid][1] for tid in topic_ids],
        mode="text",
        text=[labels[i] for i in topic_ids],
        textfont=dict(size=14, color="black"),
        showlegend=False,
        hoverinfo="skip",
    )

    fig = go.Figure([scatter, label_scatter])
    fig.update_layout(
        title=dict(text="<b>Intertopic Distance Map</b>", x=0.5, font=dict(size=24)),
        autosize=True,
        height=700,
        margin=dict(l=10, r=10, t=80, b=10),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        plot_bgcolor="white",
        hovermode="closest",
    )
    return fig.to_html(full_html=False)

# ============================================================
# MAIN TOPIC + THEME PIPELINE
# ============================================================
def generate_topic_results():
    docs = fetch_articles()
    if not docs:
        return [], {}, None, {}, {}

    # Topic model
    topic_model, topics = run_bertopic_analysis(docs)
    topic_info = topic_model.get_topic_info()
    valid_topic_ids = [t for t in topic_info.Topic if t != -1]

    # Article embeddings (used for both representative docs + themes)
    sent_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    art_emb = _normalize_rows(sent_model.encode(docs, show_progress_bar=False))

    summaries = {}
    topic_embeddings = {}

    # --- Summaries using representative docs ---
    for topic_id in valid_topic_ids:
        doc_ids = [i for i, t in enumerate(topics) if t == topic_id]
        rep_ids = get_representative_doc_ids(doc_ids, art_emb, top_k=8)
        topic_docs = [docs[i] for i in rep_ids]

        summaries[topic_id] = gpt_summarize_topic(topic_id, topic_docs)
        summaries[topic_id]["article_count"] = len(doc_ids)
        summaries[topic_id]["topic_id"] = topic_id

        topic_embeddings[topic_id] = topic_model.topic_embeddings_[topic_id].tolist()

    # --- Theme assignment (re-use art_emb) ---
    theme_texts = [f"{t}. {THEME_DESCRIPTIONS[t]}" for t in THEMES]
    theme_emb = _normalize_rows(sent_model.encode(theme_texts, show_progress_bar=False))

    theme_metrics = {t: {"volume": 0, "centrality": 0.0, "articles": set()} for t in THEMES}
    theme_metrics["Others"] = {"volume": 0, "centrality": 0.0, "articles": set()}

    for i, emb in enumerate(art_emb):
        sims = cosine_similarity([emb], theme_emb)[0]
        assigned = [THEMES[idx] for idx, score in enumerate(sims) if score >= SIMILARITY_THRESHOLD]
        if not assigned:
            assigned = ["Others"]
        for theme in assigned:
            theme_metrics[theme]["volume"] += 1
            theme_metrics[theme]["articles"].add(i)

    # Centrality
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

    for t in theme_metrics:
        theme_metrics[t].pop("centrality_raw", None)
        theme_metrics[t]["articles_raw"] = list(theme_metrics[t]["articles"])

    return docs, summaries, topic_model, topic_embeddings, theme_metrics

# ============================================================
# TEST RUN
# ============================================================
if __name__ == "__main__":
    d, s, m, e, tm = generate_topic_results()
    print("Docs:", len(d))
    print("Themes:", tm.keys())



