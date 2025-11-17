import feedparser
from newspaper import Article
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import LdaMulticore
import nltk
import re
import time
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import matplotlib.pyplot as plt

# ----------------------------
# OpenAI optimized client
# ----------------------------
from openai import OpenAI
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
NUM_TOPICS_FOR_LDA = 5


# ----------------------------
# Stopwords
# ----------------------------
try:
    nltk.data.find("corpora/stopwords")
except:
    nltk.download("stopwords")

from nltk.corpus import stopwords
stop_words = stopwords.words("english")
stop_words.extend([
    "from", "subject", "re", "edu", "use", "said", "new", "also", "one",
    "since", "per", "across"
])


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
                    articles.append({
                        "title": article.title,
                        "url": entry.link,
                        "text": article.text
                    })
            except:
                pass

            if len(articles) >= max_articles:
                return articles

    return articles


# =====================================================================
#                           LDA FUNCTIONS (unchanged)
# =====================================================================
def preprocess_texts(texts):
    processed = []
    for text in texts:
        text = re.sub(r"\S*@\S*\s?", "", text)
        text = re.sub(r"\n", " ", text)
        tokens = simple_preprocess(text, min_len=3, max_len=20)
        tokens = [w for w in tokens if w not in stop_words]
        processed.append(tokens)
    return processed


def build_lda_topic_model(texts, num_topics=10, passes=10):
    print("--- Stage 2: LDA modeling ---")
    tokenized = preprocess_texts(texts)
    dictionary = corpora.Dictionary(tokenized)
    corpus = [dictionary.doc2bow(t) for t in tokenized]

    model = LdaMulticore(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=passes,
        random_state=100,
    )
    return model, corpus, dictionary


# =====================================================================
#             📌 OPTIMIZED OPENAI TOPIC INTERPRETATION (SNIPPET BASED)
# =====================================================================

MAX_SNIPPET_CHARS = 500
MAX_COMBINED_CHARS = 9000   # safe cap for the LLM


def extract_snippet(full_text):
    """Extract 3–5 sentences as representative snippet."""
    sentences = re.split(r"(?<=[.!?]) +", full_text)
    snippet = " ".join(sentences[:5])
    return snippet[:MAX_SNIPPET_CHARS]


def get_top_docs(lda_model, corpus, article_texts, top_n=40):
    topic_docs = {t: [] for t in range(lda_model.num_topics)}

    for doc_i, bow in enumerate(corpus):
        for topic_id, prob in lda_model.get_document_topics(bow, minimum_probability=0):
            topic_docs[topic_id].append((prob, doc_i))

    for t in topic_docs:
        topic_docs[t] = sorted(topic_docs[t], reverse=True)[:top_n]

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


def interpret_topics(lda_model, corpus, article_texts, top_docs=40):
    print("\n--- Selecting representative documents ---")
    topic_docs = get_top_docs(lda_model, corpus, article_texts, top_n=top_docs)

    print("\n--- OpenAI Topic Interpretation (Snippet-Based) ---")
    results = {}
    for t, docs in topic_docs.items():
        print(f"Processing Topic {t} ...")
        results[t] = summarize_topic(t, docs, article_texts)

    return results


# =====================================================================
#                               MAIN
# =====================================================================
def main():
    start = time.time()

    # 1. Fetch
    articles = fetch_articles(MAX_ARTICLES_TO_FETCH)
    texts = [a["text"] for a in articles]

    # 2. LDA
    model, corpus, dictionary = build_lda_topic_model(
        texts, num_topics=NUM_TOPICS_FOR_LDA
    )

    print("\n=== LDA Topics ===")
    for tid, words in model.print_topics(num_words=20):
        print(f"Topic {tid}: {words}")

    # 3. Interpret topics with OpenAI
    topic_summaries = interpret_topics(model, corpus, texts)

    print("\n=== GPT Topic Summaries ===")
    for tid, summary in topic_summaries.items():
        print(f"\nTopic {tid}:\n{summary}\n")

    # 4. Visualization
    vis = gensimvis.prepare(model, corpus, dictionary)
    pyLDAvis.save_html(vis, "lda_topic_distance_map.html")
    print("Saved LDA map to lda_topic_distance_map.html")

    print(f"Total time: {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
