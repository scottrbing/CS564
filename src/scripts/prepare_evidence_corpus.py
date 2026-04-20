"""
Extract Evidence Corpus from Test Set

Pulls unique evidence documents from the MultiHopRAG test set,
matches them to the full corpus to get complete article bodies,
and outputs a focused corpus ready for graph extraction.

"""

import json
import pandas as pd
from pathlib import Path


def extract_evidence_corpus(test_set_path: str, full_corpus_path: str = None) -> pd.DataFrame:
    """
    Extract unique evidence documents from the test set.
    Deduplicates by URL, matches to full corpus for complete article bodies.
    """
    with open(test_set_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    print(f"Loaded {len(test_data)} test questions")

    # Deduplicate evidence by URL
    seen_urls = set()
    evidence_docs = []

    for item in test_data:
        for ev in item.get('evidence_list', []):
            url = ev.get('url', '')
            if url and url in seen_urls:
                continue
            if url:
                seen_urls.add(url)

            evidence_docs.append({
                'title': ev.get('title', ''),
                'source': ev.get('source', ''),
                'published_at': ev.get('published_at', ''),
                'url': url,
                'author': ev.get('author', ''),
                'category': ev.get('category', ''),
                'fact': ev.get('fact', ''),
            })

    total_refs = sum(len(item.get('evidence_list', [])) for item in test_data)
    print(f"Extracted {len(evidence_docs)} unique articles (from {total_refs} total references)")

    df_evidence = pd.DataFrame(evidence_docs)

    # Show distribution
    print(f"\nBy source publication:")
    for source, count in df_evidence['source'].value_counts().head(15).items():
        print(f"  {source}: {count}")

    print(f"\nBy category:")
    for cat, count in df_evidence['category'].value_counts().items():
        print(f"  {cat}: {count}")

    # Match to full corpus for bodies
    if full_corpus_path and Path(full_corpus_path).exists():
        df_evidence = match_full_bodies(df_evidence, full_corpus_path)
    else:
        print(f"\n[WARN] No full corpus path provided or file not found.")
        print(f"  Using 'fact' snippets as body text (graph will be less complete).")
        df_evidence['body'] = df_evidence['fact']

    return df_evidence


def match_full_bodies(df_evidence: pd.DataFrame, corpus_path: str) -> pd.DataFrame:
    """Match evidence to full corpus to get complete article bodies."""
    print(f"\nMatching to full corpus: {corpus_path}")

    # Load corpus — handle different formats
    if corpus_path.endswith('.json'):
        with open(corpus_path, 'r', encoding='utf-8') as f:
            corpus_data = json.load(f)
        df_corpus = pd.DataFrame(corpus_data if isinstance(corpus_data, list) else corpus_data.get('data', []))
    elif corpus_path.endswith('.csv'):
        df_corpus = pd.read_csv(corpus_path)
    elif corpus_path.endswith('.parquet'):
        df_corpus = pd.read_parquet(corpus_path)
    else:
        print(f"  Unknown format, skipping body matching")
        df_evidence['body'] = df_evidence['fact']
        return df_evidence

    print(f"  Full corpus: {len(df_corpus)} articles")

    # Detect body column name
    body_col = None
    for col in ['body', 'content', 'text', 'article_body']:
        if col in df_corpus.columns:
            body_col = col
            break

    if not body_col:
        print(f"  [WARN] No body column found. Columns: {list(df_corpus.columns)}")
        df_evidence['body'] = df_evidence['fact']
        return df_evidence

    # Match by URL (primary) then title (fallback)
    df_evidence['body'] = None
    matched = 0

    if 'url' in df_corpus.columns:
        url_to_body = dict(zip(df_corpus['url'], df_corpus[body_col]))
        for idx, row in df_evidence.iterrows():
            if row['url'] in url_to_body and pd.notna(url_to_body.get(row['url'])):
                df_evidence.at[idx, 'body'] = url_to_body[row['url']]
                matched += 1

    if 'title' in df_corpus.columns:
        title_to_body = dict(zip(df_corpus['title'], df_corpus[body_col]))
        for idx, row in df_evidence.iterrows():
            if pd.isna(df_evidence.at[idx, 'body']):
                if row['title'] in title_to_body and pd.notna(title_to_body.get(row['title'])):
                    df_evidence.at[idx, 'body'] = title_to_body[row['title']]
                    matched += 1

    print(f"  Matched {matched}/{len(df_evidence)} articles to full bodies")

    # Fill missing with fact snippets
    missing = df_evidence['body'].isna().sum()
    if missing > 0:
        print(f"  [WARN] {missing} articles missing bodies, using fact snippets")
        df_evidence.loc[df_evidence['body'].isna(), 'body'] = df_evidence.loc[df_evidence['body'].isna(), 'fact']

    return df_evidence


def save_corpus(df: pd.DataFrame, output_path: str):
    """Save prepared corpus."""
    df.to_json(output_path, orient='records', indent=2, force_ascii=False)
    
    body_lengths = df['body'].str.len()
    print(f"\nSaved to {output_path}")
    print(f"  {len(df)} articles, {len(df['source'].unique())} publications")
    print(f"  Body lengths: min={body_lengths.min():.0f}, median={body_lengths.median():.0f}, max={body_lengths.max():.0f}")

#  Main
if __name__ == "__main__":
    TEST_SET_PATH = "../../data/sample_queries.json"
    
    # Path to the full MultiHopRAG corpus with complete article bodies
    FULL_CORPUS_PATH = "../../data/corpus.json"
    
    OUTPUT_PATH = "../../data/evidence_corpus.json"
    # =============================================

    df = extract_evidence_corpus(
        test_set_path=TEST_SET_PATH,
        full_corpus_path=FULL_CORPUS_PATH
    )
    save_corpus(df, OUTPUT_PATH)
    
    print(f"\nNext: python graph_transformer_v2.py")
