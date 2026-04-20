"""
RAG Evaluation Script
========================================
Computes all evaluation metrics across Vector RAG, Graph RAG, and Hybrid RAG.

Metrics:
  - Accuracy (overall + per category)
  - Hallucination Rate (answering when should be Unknown)
  - Miss Rate (saying Unknown when answer exists)
  - Wrong Answer Rate (answering incorrectly, not Unknown)
  - Latency (mean, median, p95)
  - Head-to-head comparisons
  - Error type breakdown

Usage:
  python evaluate_results.py --input results/test_set.json --output results/evaluation.json
"""

import json
import pandas as pd
import numpy as np
import argparse
from collections import Counter


# Answer Normalization
def normalize(val):
    """Normalize an answer for comparison."""
    if pd.isna(val) or val is None:
        return 'unknown'
    val = str(val).strip().lower().rstrip('.')
    
    # Map empty / placeholder values
    if val in ['', 'none', 'insufficient information', 'insufficient information']:
        return 'unknown'
    
    # Single truncated letters
    if val in ['u', 'c', 'a', 's'] and len(val) == 1:
        return 'unknown'
    
    # Collapse "Yes, Yes" / "No, No" — take first
    if ',' in val:
        parts = [p.strip() for p in val.split(',')]
        if all(p in ['yes', 'no'] for p in parts):
            return parts[0]
    
    # Handle multiline "No\nYes"
    if '\n' in val:
        parts = [p.strip() for p in val.split('\n') if p.strip()]
        if all(p.lower() in ['yes', 'no'] for p in parts):
            return parts[0].lower()
    
    # Normalize synonyms
    if val in ['true', 'aligned', 'consistent']:
        return 'yes'
    
    return val


def is_correct(expected, predicted):
    """Flexible matching — handles name variants, ordering, substrings."""
    if expected == predicted:
        return True
    
    # Substring containment (e.g., "Sam Bankman-Fried" contains "Bankman-Fried")
    if expected in predicted or predicted in expected:
        return True
    
    # Name reordering (e.g., "Taylor Swift and Travis Kelce" vs "Travis Kelce, Taylor Swift")
    exp_parts = set(expected.replace(',', ' ').replace(' and ', ' ').split())
    pred_parts = set(predicted.replace(',', ' ').replace(' and ', ' ').split())
    if len(exp_parts) > 2 and exp_parts == pred_parts:
        return True
    
    return False


# Error Classification
def classify_error(expected, predicted, question_type):
    """Classify the type of error."""
    if is_correct(expected, predicted):
        return 'correct'
    
    if expected == 'unknown' and predicted != 'unknown':
        return 'hallucination'  # Answered when should be Unknown
    
    if expected != 'unknown' and predicted == 'unknown':
        return 'miss'  # Said Unknown when answer exists
    
    if expected in ['yes', 'no'] and predicted in ['yes', 'no']:
        return 'wrong_yesno'  # Got Yes/No backwards
    
    return 'wrong_answer'  # Completely wrong entity/value


# Metric Computation
def compute_metrics(df, approach_col, expected_col='expected'):
    """Compute all metrics for a single approach."""
    total = len(df)
    
    correct = df.apply(lambda r: is_correct(r[expected_col], r[approach_col]), axis=1)
    errors = df.apply(lambda r: classify_error(r[expected_col], r[approach_col], r['question_type']), axis=1)
    
    accuracy = correct.sum() / total
    
    # Error type counts
    error_counts = Counter(errors)
    hallucinations = error_counts.get('hallucination', 0)
    misses = error_counts.get('miss', 0)
    wrong_yesno = error_counts.get('wrong_yesno', 0)
    wrong_answer = error_counts.get('wrong_answer', 0)
    
    # Rates
    # Hallucination rate: among null queries, how many hallucinated?
    null_queries = df[df['question_type'] == 'null_query']
    null_total = len(null_queries)
    if null_total > 0:
        null_hallucinations = int(null_queries.apply(
            lambda r: classify_error(r[expected_col], r[approach_col], r['question_type']) == 'hallucination', axis=1
        ).sum())
    else:
        null_hallucinations = 0
    hallucination_rate = null_hallucinations / null_total if null_total > 0 else 0
    
    # Miss rate: among non-null queries, how many said Unknown?
    non_null = df[df['question_type'] != 'null_query']
    non_null_total = len(non_null)
    if non_null_total > 0:
        non_null_misses = int(non_null.apply(
            lambda r: classify_error(r[expected_col], r[approach_col], r['question_type']) == 'miss', axis=1
        ).sum())
    else:
        non_null_misses = 0
    miss_rate = non_null_misses / non_null_total if non_null_total > 0 else 0
    
    return {
        'accuracy': accuracy,
        'correct': int(correct.sum()),
        'total': total,
        'hallucination_rate': hallucination_rate,
        'hallucinations': int(null_hallucinations),
        'null_total': int(null_total),
        'miss_rate': miss_rate,
        'misses': int(non_null_misses),
        'non_null_total': int(non_null_total),
        'wrong_yesno': int(wrong_yesno),
        'wrong_answer': int(wrong_answer),
        'error_breakdown': dict(error_counts)
    }


def compute_latency_stats(df, latency_col):
    """Compute latency statistics."""
    latencies = df[latency_col].dropna()
    if len(latencies) == 0:
        return {}
    
    return {
        'mean': round(float(latencies.mean()), 2),
        'median': round(float(latencies.median()), 2),
        'p95': round(float(latencies.quantile(0.95)), 2),
        'min': round(float(latencies.min()), 2),
        'max': round(float(latencies.max()), 2),
        'std': round(float(latencies.std()), 2)
    }


def compute_head_to_head(df, col_a, col_b, expected_col='expected'):
    """Compare two approaches head-to-head."""
    a_correct = df.apply(lambda r: is_correct(r[expected_col], r[col_a]), axis=1)
    b_correct = df.apply(lambda r: is_correct(r[expected_col], r[col_b]), axis=1)
    
    a_wins = int((a_correct & ~b_correct).sum())
    b_wins = int((b_correct & ~a_correct).sum())
    both_correct = int((a_correct & b_correct).sum())
    both_wrong = int((~a_correct & ~b_correct).sum())
    
    return {
        'a_wins': a_wins,
        'b_wins': b_wins,
        'both_correct': both_correct,
        'both_wrong': both_wrong
    }


# Main Evaluation
def evaluate(input_path, output_path=None):
    """Run full evaluation and print results."""
    
    # Load data
    with open(input_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    
    print(f"Loaded {len(df)} questions from {input_path}")
    print(f"Question types: {df['question_type'].value_counts().to_dict()}")
    
    # Normalize answers
    df['expected'] = df['answer'].apply(normalize)
    df['vector'] = df['vector_rag_answer'].apply(normalize)
    df['graph'] = df['graph_rag_answer'].apply(normalize)
    df['hybrid'] = df['hybrid_rag_answer'].apply(normalize)
    
    approaches = {
        'Vector RAG': ('vector', 'vector_rag_latency'),
        'Graph RAG': ('graph', 'graph_rag_latency'),
        'Hybrid RAG': ('hybrid', 'hybrid_rag_latency')
    }
    
    results = {}
    
    # OVERALL METRICS
    print("\n" + "=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)
    
    for name, (col, lat_col) in approaches.items():
        metrics = compute_metrics(df, col)
        latency = compute_latency_stats(df, lat_col)
        
        results[name] = {'overall': metrics, 'latency': latency}
        
        print(f"\n{name}:")
        print(f"  Accuracy:          {metrics['correct']}/{metrics['total']} = {metrics['accuracy']*100:.1f}%")
        print(f"  Hallucination Rate:{metrics['hallucinations']}/{metrics['null_total']} = {metrics['hallucination_rate']*100:.1f}% (null queries)")
        print(f"  Miss Rate:         {metrics['misses']}/{metrics['non_null_total']} = {metrics['miss_rate']*100:.1f}% (said Unknown wrongly)")
        print(f"  Wrong Yes/No:      {metrics['wrong_yesno']}")
        print(f"  Wrong Entity:      {metrics['wrong_answer']}")
        if latency:
            print(f"  Latency:           mean={latency['mean']}s, median={latency['median']}s, p95={latency['p95']}s")
    
    # PER-CATEGORY METRICS
    print("\n" + "=" * 70)
    print("ACCURACY BY QUESTION TYPE")
    print("=" * 70)
    
    categories = ['inference_query', 'comparison_query', 'temporal_query', 'null_query']
    
    for qt in categories:
        subset = df[df['question_type'] == qt]
        n = len(subset)
        
        print(f"\n{qt} (n={n}):")
        print(f"  {'Approach':<15} {'Correct':>8} {'Accuracy':>10} {'Miss':>6} {'Wrong Y/N':>10} {'Wrong Ans':>10} {'Halluc':>8}")
        print(f"  {'-'*13:<15} {'-'*6:>8} {'-'*8:>10} {'-'*4:>6} {'-'*8:>10} {'-'*8:>10} {'-'*6:>8}")
        
        for name, (col, _) in approaches.items():
            metrics = compute_metrics(subset, col)
            results[name][qt] = metrics
            
            print(f"  {name:<15} {metrics['correct']:>5}/{n:<2} {metrics['accuracy']*100:>9.1f}% "
                  f"{metrics['misses']:>5} {metrics['wrong_yesno']:>9} {metrics['wrong_answer']:>9} "
                  f"{metrics['hallucinations']:>7}")
    
    # PER-CATEGORY LATENCY
    print("\n" + "=" * 70)
    print("LATENCY BY QUESTION TYPE (mean seconds)")
    print("=" * 70)
    
    print(f"\n  {'Category':<20} {'Vector':>10} {'Graph':>10} {'Hybrid':>10}")
    print(f"  {'-'*18:<20} {'-'*8:>10} {'-'*8:>10} {'-'*8:>10}")
    
    for qt in categories:
        subset = df[df['question_type'] == qt]
        v_lat = subset['vector_rag_latency'].dropna().mean()
        g_lat = subset['graph_rag_latency'].dropna().mean()
        h_lat = subset['hybrid_rag_latency'].dropna().mean()
        
        print(f"  {qt:<20} {v_lat:>9.2f}s {g_lat:>9.2f}s {h_lat:>9.2f}s")
        
        results.setdefault('latency_by_type', {})[qt] = {
            'vector': round(v_lat, 2) if not pd.isna(v_lat) else None,
            'graph': round(g_lat, 2) if not pd.isna(g_lat) else None,
            'hybrid': round(h_lat, 2) if not pd.isna(h_lat) else None
        }
    
    # HEAD-TO-HEAD
    print("\n" + "=" * 70)
    print("HEAD-TO-HEAD COMPARISONS")
    print("=" * 70)
    
    matchups = [
        ('Graph RAG vs Vector RAG', 'graph', 'vector'),
        ('Hybrid RAG vs Vector RAG', 'hybrid', 'vector'),
        ('Hybrid RAG vs Graph RAG', 'hybrid', 'graph'),
    ]
    
    for label, col_a, col_b in matchups:
        h2h = compute_head_to_head(df, col_a, col_b)
        results[label] = h2h
        
        a_name, b_name = label.split(' vs ')
        print(f"\n{label}:")
        print(f"  {a_name} wins:  {h2h['a_wins']}")
        print(f"  {b_name} wins: {h2h['b_wins']}")
        print(f"  Both correct:     {h2h['both_correct']}")
        print(f"  Both wrong:       {h2h['both_wrong']}")
        
        # Per category
        for qt in categories:
            subset = df[df['question_type'] == qt]
            h2h_qt = compute_head_to_head(subset, col_a, col_b)
            print(f"    {qt}: {a_name} +{h2h_qt['a_wins']}, {b_name} +{h2h_qt['b_wins']}, both={h2h_qt['both_correct']}")
    
    # ERROR ANALYSIS
    print("\n" + "=" * 70)
    print("DETAILED ERROR ANALYSIS")
    print("=" * 70)
    
    for name, (col, _) in approaches.items():
        print(f"\n{name} errors by type and category:")
        for qt in categories:
            subset = df[df['question_type'] == qt]
            errors = subset.apply(
                lambda r: classify_error(r['expected'], r[col], r['question_type']), axis=1
            )
            error_counts = Counter(errors)
            error_counts.pop('correct', None)
            if error_counts:
                print(f"  {qt}: {dict(error_counts)}")
    
    # QUESTIONS WHERE ALL THREE FAIL
    print("\n" + "=" * 70)
    print("UNIVERSALLY HARD QUESTIONS (all three wrong)")
    print("=" * 70)
    
    v_correct = df.apply(lambda r: is_correct(r['expected'], r['vector']), axis=1)
    g_correct = df.apply(lambda r: is_correct(r['expected'], r['graph']), axis=1)
    h_correct = df.apply(lambda r: is_correct(r['expected'], r['hybrid']), axis=1)
    
    all_wrong = df[~v_correct & ~g_correct & ~h_correct]
    print(f"\nTotal: {len(all_wrong)} questions")
    for qt in categories:
        cnt = len(all_wrong[all_wrong['question_type'] == qt])
        if cnt > 0:
            print(f"  {qt}: {cnt}")
    
    # Show a few examples
    for _, r in all_wrong.head(5).iterrows():
        print(f"\n  [{r['question_type']}]")
        print(f"  Expected: {r['expected']}")
        print(f"  Vector: {r['vector']}, Graph: {r['graph']}, Hybrid: {r['hybrid']}")
        print(f"  Q: {r['query'][:120]}...")
    
    # UNIQUE WINS (where only one approach got it right)
    print("\n" + "=" * 70)
    print("UNIQUE WINS (only one approach correct)")
    print("=" * 70)
    
    only_vector = df[v_correct & ~g_correct & ~h_correct]
    only_graph = df[~v_correct & g_correct & ~h_correct]
    only_hybrid = df[~v_correct & ~g_correct & h_correct]
    
    print(f"Only Vector correct: {len(only_vector)}")
    print(f"Only Graph correct:  {len(only_graph)}")
    print(f"Only Hybrid correct: {len(only_hybrid)}")
    
    for label, subset in [("Only Graph", only_graph), ("Only Hybrid", only_hybrid)]:
        if len(subset) > 0:
            print(f"\n{label} wins:")
            for _, r in subset.iterrows():
                print(f"  [{r['question_type']}] Expected: {r['expected']}")
    
    # SUMMARY TABLE (copy-paste for thesis)
    print("\n" + "=" * 70)
    print("SUMMARY TABLE (for thesis)")
    print("=" * 70)
    
    print(f"\n{'Metric':<30} {'Vector RAG':>12} {'Graph RAG':>12} {'Hybrid RAG':>12}")
    print(f"{'-'*28:<30} {'-'*10:>12} {'-'*10:>12} {'-'*10:>12}")
    
    for name, (col, lat_col) in approaches.items():
        m = results[name]['overall']
    
    # Print summary rows
    v_m = results['Vector RAG']['overall']
    g_m = results['Graph RAG']['overall']
    h_m = results['Hybrid RAG']['overall']
    
    v_l = results['Vector RAG']['latency']
    g_l = results['Graph RAG']['latency']
    h_l = results['Hybrid RAG']['latency']
    
    rows = [
        ('Overall Accuracy', f"{v_m['accuracy']*100:.1f}%", f"{g_m['accuracy']*100:.1f}%", f"{h_m['accuracy']*100:.1f}%"),
        ('Inference Accuracy', 
         f"{results['Vector RAG'].get('inference_query',{}).get('accuracy',0)*100:.1f}%",
         f"{results['Graph RAG'].get('inference_query',{}).get('accuracy',0)*100:.1f}%",
         f"{results['Hybrid RAG'].get('inference_query',{}).get('accuracy',0)*100:.1f}%"),
        ('Comparison Accuracy',
         f"{results['Vector RAG'].get('comparison_query',{}).get('accuracy',0)*100:.1f}%",
         f"{results['Graph RAG'].get('comparison_query',{}).get('accuracy',0)*100:.1f}%",
         f"{results['Hybrid RAG'].get('comparison_query',{}).get('accuracy',0)*100:.1f}%"),
        ('Temporal Accuracy',
         f"{results['Vector RAG'].get('temporal_query',{}).get('accuracy',0)*100:.1f}%",
         f"{results['Graph RAG'].get('temporal_query',{}).get('accuracy',0)*100:.1f}%",
         f"{results['Hybrid RAG'].get('temporal_query',{}).get('accuracy',0)*100:.1f}%"),
        ('Null Query Accuracy',
         f"{results['Vector RAG'].get('null_query',{}).get('accuracy',0)*100:.1f}%",
         f"{results['Graph RAG'].get('null_query',{}).get('accuracy',0)*100:.1f}%",
         f"{results['Hybrid RAG'].get('null_query',{}).get('accuracy',0)*100:.1f}%"),
        ('Hallucination Rate', 
         f"{v_m['hallucination_rate']*100:.1f}%", f"{g_m['hallucination_rate']*100:.1f}%", f"{h_m['hallucination_rate']*100:.1f}%"),
        ('Miss Rate',
         f"{v_m['miss_rate']*100:.1f}%", f"{g_m['miss_rate']*100:.1f}%", f"{h_m['miss_rate']*100:.1f}%"),
        ('Mean Latency',
         f"{v_l.get('mean','N/A')}s", f"{g_l.get('mean','N/A')}s", f"{h_l.get('mean','N/A')}s"),
        ('P95 Latency',
         f"{v_l.get('p95','N/A')}s", f"{g_l.get('p95','N/A')}s", f"{h_l.get('p95','N/A')}s"),
    ]
    
    for row_name, v, g, h in rows:
        print(f"{row_name:<30} {v:>12} {g:>12} {h:>12}")
    
    # Save results
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {output_path}")
    
    return results


# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate RAG results')
    parser.add_argument('--input', type=str, default='./results/test_set.json',
                        help='Path to test set JSON with results')
    parser.add_argument('--output', type=str, default='./results/evaluation.json',
                        help='Path to save evaluation metrics')
    args = parser.parse_args()
    
    evaluate(args.input, args.output)