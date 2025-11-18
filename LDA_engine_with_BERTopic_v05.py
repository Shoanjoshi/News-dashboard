# ============================
# LDA_engine_with_BERTopic_v05.py
# Version 5.1 – Stabilized topic extraction
# ============================

import feedparser
from newspaper import Article
from bertopic import BERTopic
from openai import OpenAI

# OpenAI Client
client = OpenAI()

# ------------ RSS Fetcher ---------------
def fetch_articles(rss_urls):
    """Fetches news articles from RSS feeds."""
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
                    # Ensure sufficient content
                    if len(art.text) > 300:
                        articles.append(art.text)
                except:
                    # Ignore failures
                    pass
        except:
            pass
    return articles


# ------------ Topic Model (Stable v5.1) ---------------
def run_topic_model(texts):
    """
    Version 5.1 topic model settings:
    - Let BERTopic automatically determine number of topics.
    - Enforce cluster minimum size for stability.
    - Use probabilities to maintain separation quality.
    """

    topic_model = BERTopic(
        language="english",
        min_topic_size=5,            # Helps prevent collapsing into 1 topic
        calculate_probabilities=True,
        nr_topics=None,              # Allow natural clustering
        verbose=True
    )

    topics, _ = topic_model.fit_transform(texts)
    return topic_model, topics


# ------------ GPT Topic Summaries --------
def summarize_topic_gpt(topic_id, top_words, sample_docs):
    """
    Generate a human-readable topic label + summary using GPT.
    Output format: JSON with keys title and summary.
    """

    prompt = f"""
You are an expert news analyst.

Top words for topic {topic_id}:
{top_words}

Example documents:
{sample_docs[:2]}

Give a short topic name (max 6 words) and a 2–3 sentence explanation.

Return JSON format:
{{ "title": "...", "summary": "..." }}
"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    # 🔧 FIX APPLIED – Correct way to access GPT output
    return response.choices[0].message.content
