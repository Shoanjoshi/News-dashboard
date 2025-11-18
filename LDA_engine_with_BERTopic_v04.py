# ============================
# LDA_engine_with_BERTopic_v05.py
# Restored stable Version 5
# ============================

import feedparser
import requests
from newspaper import Article
from bertopic import BERTopic
from openai import OpenAI

client = OpenAI()

# ------------ RSS Fetcher ---------------
def fetch_articles(rss_urls):
    articles = []
    for url in rss_urls:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            link = entry.link
            try:
                art = Article(link)
                art.download()
                art.parse()
                if len(art.text) > 300:  # ensures enough content
                    articles.append(art.text)
            except:
                pass
    return articles


# ------------ Topic Model ----------------
def run_topic_model(texts, nr_topics=5):
    """
    Version 5 logic:
    - Use BERTopic default behavior
    - DO NOT override UMAP
    - DO NOT force min_topic_size
    - DO NOT force fixed topic count
    """

    topic_model = BERTopic(
        language="english",
        calculate_probabilities=False,
        nr_topics=nr_topics
    )

    topics, _ = topic_model.fit_transform(texts)
    return topic_model, topics


# ------------ GPT Topic Summaries --------
def summarize_topic_gpt(topic_id, top_words, sample_docs):
    """
    GPT topic labeling (stable).
    """

    prompt = f"""
You are an expert news analyst.

Here are the top words for a topic:
{top_words}

Here are sample documents:
{sample_docs[:2]}

Give a short name for this topic (max 6 words), 
and a 2–3 sentence explanation of what the topic is about.

Return in JSON with keys: title, summary.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    return response.choices[0].message["content"]
