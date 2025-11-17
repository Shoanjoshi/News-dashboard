import os
import json
import datetime

from LDA_engine_with_OPENAI_v03 import (
    fetch_articles,
    build_lda_topic_model,
    interpret_topics,
    MAX_ARTICLES_TO_FETCH,
    NUM_TOPICS_FOR_LDA
)

import pyLDAvis
import pyLDAvis.gensim_models as gensimvis



OUTPUT_DIR = "dashboard"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_dashboard():

    # ------------------------------------------------------------
    # 1. Run the pipeline
    # ------------------------------------------------------------
    print("Fetching articles...")
    articles = fetch_articles(MAX_ARTICLES_TO_FETCH)
    texts = [a["text"] for a in articles]

    print("Running LDA...")
    lda_model, corpus, dictionary = build_lda_topic_model(
        texts, num_topics=NUM_TOPICS_FOR_LDA
    )

    print("Generating LLM topic summaries...")
    topic_summaries = interpret_topics(lda_model, corpus, texts)


    # ------------------------------------------------------------
    # 2. Build LDA HTML and embed it
    # ------------------------------------------------------------
    lda_html_path = os.path.join(OUTPUT_DIR, "lda_vis.html")

    vis = gensimvis.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(vis, lda_html_path)

    # Read but remove outer HTML/BODY tags so embedding looks clean
    with open(lda_html_path, "r", encoding="utf-8") as f:
        lda_raw_html = f.read()

    # Remove <html>, <body>
    lda_clean = (
        lda_raw_html
        .replace("<html>", "")
        .replace("</html>", "")
        .replace("<body>", "")
        .replace("</body>", "")
    )


    # ------------------------------------------------------------
    # 3. Build final dashboard page
    # ------------------------------------------------------------
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    final_html = f"""
<html>
<head>
    <title>Daily News Topic Dashboard</title>

    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
        }}

        .header {{
            background: #f2f2f2;
            padding: 12px;
            font-size: 15px;
            font-weight: bold;
            border-bottom: 2px solid #ccc;
        }}

        .main-container {{
            display: flex;
            height: calc(100vh - 48px);
        }}

        .left-panel {{
            width: 55%;
            padding: 10px;
            border-right: 3px solid #ccc;
            overflow-y: scroll;
        }}

        .right-panel {{
            width: 45%;
            padding: 20px;
            overflow-y: scroll;
        }}

        .topic-block {{
            background: #fafafa;
            margin-bottom: 25px;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #ddd;
        }}

        h2 {{
            margin-top: 0;
        }}
    </style>

</head>

<body>

<div class="header">
    Dashboard updated: {timestamp}
</div>

<div class="main-container">

    <!-- LEFT SIDE: LDA MAP -->
    <div class="left-panel">
        <h2>LDA Topic Map</h2>
        {lda_clean}
    </div>

    <!-- RIGHT SIDE: GPT TOPIC SUMMARIES -->
    <div class="right-panel">
        <h2>Topic Summaries (GPT)</h2>
"""

    # Append each topic summary in HTML
    for tid, summary_json in topic_summaries.items():
        data = json.loads(summary_json)

        final_html += f"""
        <div class="topic-block">
            <h3>Topic {tid}: {data['label']}</h3>

            <p>{data['description']}</p>

            <b>Subthemes:</b>
            <ul>
                {''.join(f"<li>{st}</li>" for st in data['subthemes'])}
            </ul>
        </div>
        """

    # Close final HTML
    final_html += """
    </div>
</div>

</body>
</html>
"""

    # ------------------------------------------------------------
    # 4. Save dashboard
    # ------------------------------------------------------------
    output_file = os.path.join(OUTPUT_DIR, "index.html")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(final_html)

    print(f"Dashboard created → {output_file}")


if __name__ == "__main__":
    generate_dashboard()

