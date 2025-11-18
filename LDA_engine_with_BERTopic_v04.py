import feedparser
from newspaper import Article
import nltk
import ssl
import requests
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import json
import time

# ------------------------------------------------------------
# Fix B: Fetch more articles → more stable BERTopic clustering
# ------------------------------------------------------------
MAX_ARTICLES_TO_FETCH = 150     # was 60
NUM_TOPICS_FOR_LDA = 12         # used only for LLM summaries

# ------------------------------------------------------------
# OpenAI client initialization
# ------------------------------------------------------------
client = OpenAI()

# Fix for SSL (newspaper3k sometimes fails without this)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download("punkt", quiet=True)


# ------------------------------------------------------------
# Fetch & parse RSS feeds
# ------------------------------------------------------------
RSS_FEEDS = [
    "http://feeds.bbci.co.uk/news/rss.xml",
    "https://rss.cnn.com/rss/edition.rss",
    "https://feeds.a.dj.com/rss/RSSWorldNews.xml",
    "https://www.reutersagency.com/feed/?best-sectors=world&post_type=best",
]


def fetch_articles(max_articles=MAX_ARTICLES_TO_FETCH):
    articles = []
    for feed_url in RSS_FEEDS:
        print(f"Fetching RSS feed: {feed_url}")
        feed = feedparser.parse(feed_url)

        for entry in feed.entries[:50]:
            url = entry.get("link")
            if not url:
                continue

            try:
                art = Article(url)
                art.download()
                art.parse()

                if len(art.text.strip()) < 200:
                    continue

                articles.append({
                    "title": art.title,
                    "text": art.text,
                    "url": url
                })

                if len(articles) >= max_articles:
                    return articles

            except Exception:
                continue

    return articles


# ------------------------------------------------------------
# Fix C: Let BERTopic choose the optimal number of topics
# ------------------------------------------------------------
def build_bertopic_model(texts):
    # Use a more stable embedding model
    embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    topic_model = BERTopic(
        embedding_model=embedding_model,
        verbose=True
    )

    topics, probs = topic_model.fit_transform(texts)
    return topic_model, topics, probs


# ------------------------------------------------------------
# GPT-based topic interpretation
# ------------------------------------------------------------
TOPIC_PROMPT = """
You are an expert analyst. The following text documents belong to Topic {topic_id}.
Your task:
1. Assign a short, human-readable label to this topic (5 words or fewer).
2. Write a concise 2–3 sentence description of the topic.
3. List 2–4 subthemes in bullet form.

Respond ONLY with valid JSON using keys:
label, description, subthemes
"""


def interpret_topics(topic_model, texts):
    topic_info = topic_model.get_topic_info()
    doc_info = topic_model.get_document_info(texts)

    valid_topics = [
        t for t in topic_info["Topic"].tolist()
        if t != -1
    ]

    summaries = {}

    for tid in valid_topics:
        print(f"Processing Topic {tid} ...")

        df_t = (
            doc_info[doc_info["Topic"] == tid]
            .sort_values("Probability", ascending=False)
            .head(10)
        )

        docs = []
        for idx in df_t.index:
            docs.append(texts[idx][:1000])

        joined_text = "\n\n".join(docs)

        prompt = TOPIC_PROMPT.format(topic_id=tid) + "\n\nDocuments:\n" + joined_text

        success = False
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.4,
                )
                summaries[tid] = response.choices[0].message.content
                success = True
                break
            except Exception as e:
                print(f"Retrying topic {tid} due to error: {e}")
                time.sleep(2)

        if not success:
            summaries[tid] = json.dumps({
                "label": f"Topic {tid}",
                "description": "Summary generation failed.",
                "subthemes": []
            })

    return summaries
