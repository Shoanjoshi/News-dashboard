# ============================
# LDA_engine_with_BERTopic_v05.py
# Version 5.1 – stabilized topic generation
# ============================

import feedparser
from newspaper import Article
from bertopic import BERTopic
from openai import OpenAI

client = OpenAI()

# ------------ RSS Fetcher ---------------
def fetch_articles(rss_urls):
    articles = []
    for url in rss_urls:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                link = entry.link
                try:
                    art = Article(link)
                    art.download()
                    art.parse()
                    if len(art.text) > 300:
                        articles.append(art.text)
                except:
                    pass
        except:
            pass
    return articles


# ------------ Topic Model (Stable v5.1) ---------------
def run_topic_model(texts):
    """
    Version 5.1 topic model settings:
    - Allow BERTopic to naturally determine number of topics
    - Force clusters of at least 5 docs
    - Calculate probabilities for cleaner separation
    - DO NOT set nr_topics=5 (causes reduction errors)
    """

    topic_model = BERTopic(
        language="english",
        min_topic_size=5,             # Ensures diversity → stable topic count
        calculate_probabilities=True, # Helps topic formation
        nr_topics=None,               # Let model discover #topics instead of forcing
        verbose=True
    )

    topics, _ = topic_model.fit_transform(texts)
    return topic_model, topics


# ------------ GPT Topic Summaries --------
def summarize_topic_gpt(topic_id, top_words, sample_docs):
    prompt = f"""
You are a senior news analyst.

Top words for topic {topic_id}:
{top_words}

Example documents:
{sample_docs[:2]}

Give:
1. A short title (max 6 words)
2. A 2–3 sentence explanation

Return in JSON with keys: title, summary.
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    return response.choices[0].message["content"]
