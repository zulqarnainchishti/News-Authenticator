import os
import json
import pandas as pd
import re
from nltk.tokenize import sent_tokenize

DATASET = "C:/Users/ASP/OneDrive/Documents/News Authenticity Verifier/data/raw"
DATA_PATH = "C:/Users/ASP/OneDrive/Documents/News Authenticity Verifier/data/articles.parquet"

def clean_text(text):
    # Step 1: Remove URLs, hyperlinks, markdown-style links
    text = re.sub(r"http\S+|www\.\S+|\[.*?\]\(.*?\)", " ", text)

    # Step 2: Remove emojis, special symbols, and non-ASCII
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    # Step 3: Remove boilerplate UI terms or sections (common in feed aggregations)
    boilerplate_patterns = [
        r"YOU MAY ALSO LIKE",
        r"RELATED TOPICS",
        r"COMMENTS",
        r"FOLLOWERS",
        r"EDITORS' PICKS",
        r"SEE ALL TRENDS",
        r"BY [A-Z].*?,",         # bylines like "By Editors
        r"\buser\b|\bpublisher\b|\bwebsite\b",
        r"\[\s*·.*?followers\s*\]",
        r"\s*·\s*\w+\s*·",       # " · publisher · "
        r"trending", r"trend", r"see all",
    ]
    for pattern in boilerplate_patterns:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

    # Step 4: Collapse whitespace and newlines
    text = re.sub(r"\s+", " ", text).strip()

    # Step 5: Sentence tokenization
    sentences = sent_tokenize(text)

    # Step 6: Filter out meaningless or short lines (< 6 words)
    meaningful = [
        s for s in sentences
        if len(s.split()) >= 6
        and not re.match(r"^[·\-–\s]+$", s)
        and not any(x in s.lower() for x in ["followers", "click", "share", "subscribe"])
    ]

    # Step 7: Join into coherent paragraph
    cleaned_text = " ".join(meaningful)

    return cleaned_text.strip()

def collect_articles(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        article = json.load(f)
    
    if article.get("thread", {}).get("country") != "US":
        return None
    
    published = article.get("published", "")[:10]  # YYYY-MM-DD
    categories = article.get("categories", [])

    persons = [p.get("name", "") for p in article.get("entities", {}).get("persons", [])]
    organizations = [o.get("name", "") for o in article.get("entities", {}).get("organizations", [])]
    locations = [l.get("name", "") for l in article.get("entities", {}).get("locations", [])]
    entities = [e for e in (persons + organizations + locations) if e]

    title = clean_text(article.get("title", ""))
    content = clean_text(article.get("text", ""))

    if not title or not content:
        return None

    return {
        "published": published,
        "categories": categories,
        "entities": entities,
        "title": title,
        "content": content
    }

if __name__ == "__main__":
    articles_data = []
    
    for folder in os.listdir(DATASET):
        folder_path = os.path.join(DATASET, folder)

        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            
            articles_data.append(collect_articles(file_path))

            print(f"📄Processed file: {file}")
        print(f"📁Processed folder: {folder}")

    df = pd.DataFrame(articles_data)
    print(f"\nTotal articles: {len(df)}")

    df.to_parquet(DATA_PATH, index=False, engine="pyarrow")
    print(f"\nCleaned dataset saved to: {DATA_PATH}")