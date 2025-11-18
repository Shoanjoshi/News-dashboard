import feedparser
from newspaper import Article
import re
import time
import nltk

from bertopic import BERTopic
from openai import OpenAI

# ----------------------------
# OpenAI optimized client
# ----------------------------
client = OpenAI()  # Reads your OPENAI_API_KEY from environment


# ---- Configuration ----
RSS_FEEDS = [
    # 1) REUTERS
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.reuters.com/reuters/worldNews",
    "https://feeds.reuters.com/reuters/USbusinessNews",
    "https://feeds.reuters.com/reuters/technologyNews",
    "https://feeds.reuters.com/reuters/environment",
    "https://feeds.reuters.com/reuters/politicsNews",
    "https://feeds.reuters.com/reuters/companyNews",

    # 2) AP NEWS
    "https://apnews.com/apf-news",
    "https://apnews.com/apf-topnews",
    "https://apnews.com/apf-business",
    "https://apnews.com/apf-worldnews",
    "https://apnews.com/apf-technology",

    # 3) BBC
    "http://feeds.bbci.co.uk/news/world/rss.xml",
    "http://feeds.bbci.co.uk/news/business/rss.xml",
    "http://feeds.bbci.co.uk/news/technology/rss.xml",
    "http://feeds.bbci.co.uk/news/politics/rss.xml",
    "http://feeds.bbci.co.uk/news/science_and_environment/rss.xml",

    # 4) CNN
    "http://rss.cnn.com/rss/edition_world.rss",
    "http://rss.cnn.com/rss/edition_business.rss",
    "http://rss.cnn.com/rss/edition_technology.rss",
    "http://rss.cnn.com/rss/money_latest.rss",

    # 5) MarketWatch
    "https://feeds.marketwatch.com/marketwatch/topstories/",
    "https://feeds.marketwatch.com/marketwatch/marketpulse/",
    "https://feeds.marketwatch.com/marketwatch/realtimeheadlines/",

    # 6) CNBC
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "https://www.cnbc.com/id/10001147/device/rss/rss.html",
    "https://www.cnbc.com/id/15839069/device/rss/rss.html",
    "https://www.cnbc.com/id/10000664/device/rss/rss.html",
    "https://www.cnbc.com/id/100727362/device/rss/rss.html",

    # 7) Guardian
    "https://www.theguardian.com/world/rss",
    "https://www.theguardian.com/business/rss",
    "https://www.theguardian.com/commentisfree/rss",
    "https://www.theguardian.com/environment/rss",
    "https://www.theguardian.com/us-news/rss",

    # 8) Yahoo Finance
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC",
    "https://www.yahoo.com/news/rss",
    "https://finance.yahoo.com/news/rssindex",
]

MAX_ARTICLES_TO_FETCH = 500
NUM_TOPICS_FOR_LDA = 5  # kept same name for compatibility with previous code


# ----------------------------
# Stopwords (optional, mainly for snippet cleaning)
# ----------------------------
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

from nltk.corpus import stopwords

stop_words = stopwords.words("english")
stop_words.extend(
    [
        "from",
        "subject",
        "re",
        "edu",
        "use",
        "said",
        "new",
        "also",
        "one",
        "since",
        "per",
        "across",
    ]
)


# =====================================================================
#                NEWS SCRAPING FUNCTIONS (unchanged)
# =====================================================================
def fetch_articles(max_articles):
    articles = []
    print(f"--- Stage 1: Fetching up to {max_articles} articles ---")

    for feed_url in RSS_FEEDS:
        feed = feedparser.parse(feed_url)
        per_feed_limit = max_articles // len(RSS_FEEDS)

        for entry in feed.entries[:per_feed_limit]:
            try:
                article = Article(entry.link)
                article.download()
                article.parse()
                if len(article.text) > 300:
                    articles.append(
                        {
                            "title": article.title,
                            "url": entry.link,
                            "text": article.text,
                        }
                    )
            except Exception:
                pass

            if len(articles) >= max_articles:
                return articles

    return articles


# =====================================================================
#                  BERTOPIC TOPIC MODELING (FIX 6)
# =====================================================================

def build_bertopic_model(texts, num_topics=10):
    """
    Build a BERTopic model over the raw article texts.

    We keep the name NUM_TOPICS_FOR_LDA for compatibility, but
    internally we use BERTopic with nr_topics=num_topics.
    """
    print("--- Stage 2: BERTopic modeling ---")

    # BERTopic will create embeddings internally (default uses a sentence-transformer)
    topic_model = BERTopic(
        nr_topics=num_topics,
        calculate_probabilities=True,
        verbose=True,
    )

    topics, probs = topic_model.fit_transform(texts)
    return topic_model, topics, probs


# =====================================================================
#             GPT TOPIC INTERPRETATION (SNIPPET-BASED)
# =====================================================================

MAX_SNIPPET_CHARS = 500
MAX_COMBINED_CHARS = 9000  # safety cap for the LLM


def extract_snippet(full_text):
    """Extract 3–5 sentences as representative snippet."""
    sentences = re.split(r"(?<=[.!?]) +", full_text)
    snippet = " ".join(sentences[:5])
    return snippet[:MAX_SNIPPET_CHARS]


def get_top_docs_bertopic(topic_model, texts, top_n=40):
    """
    Use BERTopic's document info to get top documents per topic
    based on the per-document topic probability.

    Returns:
        dict[topic_id] = list of (probability, doc_index)
    """
    print("--- Selecting representative documents (BERTopic) ---")
    info_df = topic_model.get_document_info(texts)

    topic_docs = {}
    for topic_id in info_df["Topic"].unique():
        if topic_id == -1:
            # -1 is BERTopic's 'outlier' topic
            continue

        df_topic = (
            info_df[info_df["Topic"] == topic_id]
            .sort_values("Probability", ascending=False)
            .head(top_n)
        )
        topic_docs[topic_id] = list(zip(df_topic["Probability"], df_topic.index))

    return topic_docs


def summarize_topic(topic_id, docs, article_texts, max_docs=12):
    """
    SINGLE LLM CALL PER TOPIC.
    Uses snippet-based extraction for accuracy and stability.
    """
    snippets = []
    for _, doc_idx in docs[:max_docs]:
        snippet = extract_snippet(article_texts[doc_idx])
        snippets.append(snippet)

    combined = "\n\n------\n\n".join(snippets)
    combined = combined[:MAX_COMBINED_CHARS]  # safety cap

    prompt = f"""
You are an expert topic analyst.

Below are multiple short text snippets representing Topic {topic_id}:

{combined}

Your tasks:
1. Describe what this topic cluster is about in 2–3 sentences.
2. Provide a topic label in 3–5 words.
3. Provide 2–3 subthemes.

Return ONLY valid JSON:
{{
  "topic_id": {topic_id},
  "label": "",
  "description": "",
  "subthemes": []
}}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    return response.choices[0].message.content


def interpret_topics(topic_model, texts, top_docs=40):
    """
    Build topic → GPT summary mapping using BERTopic clusters.
    """
    topic_docs = get_top_docs_bertopic(topic_model, texts, top_n=top_docs)

    print("\n--- OpenAI Topic Interpretation (BERTopic-based) ---")
    results = {}
    for t, docs in topic_docs.items():
        print(f"Processing Topic {t} ...")
        results[t] = summarize_topic(t, docs, texts)

    return results


# =====================================================================
#                               MAIN (optional local test)
# =====================================================================
def main():
    start = time.time()

    # 1. Fetch
    articles = fetch_articles(MAX_ARTICLES_TO_FETCH)
    texts = [a["text"] for a in articles]

    # 2. BERTopic
    topic_model, topics, probs = build_bertopic_model(
        texts, num_topics=NUM_TOPICS_FOR_LDA
    )

    print("\n=== BERTopic Topics ===")
    print(topic_model.get_topic_info())

    # 3. Interpret topics with OpenAI
    topic_summaries = interpret_topics(topic_model, texts)

    print("\n=== GPT Topic Summaries ===")
    for tid, summary in topic_summaries.items():
        print(f"\nTopic {tid}:\n{summary}\n")

    # 4. Visualization (for local testing)
    vis_fig = topic_model.visualize_topics()
    vis_fig.write_html("bertopic_topic_map.html")
    print("Saved BERTopic map to bertopic_topic_map.html")

    print(f"Total time: {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
