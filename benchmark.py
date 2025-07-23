#!/usr/bin/env python3
"""
Benchmark Suite: Adversarial Fake News & Misinformation NLP Classifier
Evaluates TF-IDF + Logistic Regression (L2), Passive-Aggressive Classifier, and
Complement Naive Bayes using 5-Fold Stratified Cross-Validation on a synthesized
adversarial news corpus with lexical, syntactic, and sensationalist token patterns.
"""

import json
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline


def generate_adversarial_news_corpus(n_samples: int = 1200, random_state: int = 42):
    """
    Synthesizes a domain-calibrated corpus of reliable journalism vs. adversarial
    disinformation articles, incorporating realistic lexical overlap and noise.
    """
    rng = np.random.default_rng(random_state)

    reliable_tokens = [
        "reuters", "associated press", "bipartisan", "committee report", "peer reviewed",
        "clinical trial", "statistical agency", "central bank", "quarterly filing",
        "official statement", "regulatory framework", "parliamentary session",
        "empirical analysis", "audit findings", "diplomatic summit", "cbo projection",
        "epidemiological data", "federal reserve", "supreme court ruling", "economic indicator"
    ]
    disinfo_tokens = [
        "shocking coverup", "secret cabal", "mainstream media blackout", "miracle cure",
        "leaked bombshell", "unverified whistleblower", "deep state plot", "urgent warning",
        "banned video", "doctors hate this", "anonymous patriot", "hoax exposed",
        "total collapse imminent", "hidden agenda", "shadow syndicate", "wake up",
        "suppressed truth", "globalist directive", "rigged narrative", "viral threads"
    ]
    shared_tokens = [
        "administration", "election", "policy", "market", "public health", "investigation",
        "international", "senate", "energy sector", "technology", "conference", "citizen",
        "legislation", "budget", "security", "department", "spokesperson", "announcement"
    ]

    texts, labels = [], []
    for i in range(n_samples):
        is_fake = int(i % 2 == 1)
        # Introduce 9% adversarial camouflage where fake articles mimic reliable vocabulary
        adversarial_camouflage = rng.random() < 0.09
        doc_words = []

        # Base shared context
        doc_words.extend(rng.choice(shared_tokens, size=rng.integers(8, 16), replace=True))

        if (is_fake and not adversarial_camouflage) or (not is_fake and adversarial_camouflage):
            doc_words.extend(rng.choice(disinfo_tokens, size=rng.integers(4, 10), replace=True))
            doc_words.extend(rng.choice(reliable_tokens, size=rng.integers(1, 5), replace=True))
        else:
            doc_words.extend(rng.choice(reliable_tokens, size=rng.integers(4, 10), replace=True))
            doc_words.extend(rng.choice(disinfo_tokens, size=rng.integers(1, 5), replace=True))

        rng.shuffle(doc_words)
        texts.append(" ".join(doc_words))
        labels.append(is_fake)

    return np.array(texts), np.array(labels)


def render_svg_chart(results: list, output_path: Path):
    """Generates a dark-mode vector performance chart for NLP classification metrics."""
    width, height = 860, 430
    metrics = ["roc_auc", "f1", "accuracy", "precision"]
    metric_labels = {
        "roc_auc": "ROC-AUC",
        "f1": "F1-Score",
        "accuracy": "Accuracy",
        "precision": "Precision",
    }
    colors = {
        "roc_auc": "#38bdf8",
        "f1": "#a78bfa",
        "accuracy": "#34d399",
        "precision": "#f472b6",
    }

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%">',
        '  <defs>',
        '    <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">',
        '      <stop offset="0%" stop-color="#090d16"/>',
        '      <stop offset="100%" stop-color="#1e293b"/>',
        '    </linearGradient>',
        '  </defs>',
        f'  <rect width="{width}" height="{height}" rx="12" fill="url(#bgGrad)" stroke="#334155" stroke-width="1.5"/>',
        '  <text x="32" y="42" fill="#f8fafc" font-family="Segoe UI, Inter, sans-serif" font-size="20" font-weight="700">'
        'Adversarial Fake News &amp; Misinformation NLP Classifier — 5-Fold CV</text>',
        '  <text x="32" y="66" fill="#94a3b8" font-family="Segoe UI, Inter, sans-serif" font-size="13">'
        'Sublinear TF-IDF (1–2 n-grams) | N = 1,200 Articles (9% Adversarial Camouflage)</text>',
    ]

    lx = 32
    for m in metrics:
        svg.append(f'  <rect x="{lx}" y="84" width="12" height="12" rx="3" fill="{colors[m]}"/>')
        svg.append(
            f'  <text x="{lx + 18}" y="94" fill="#cbd5e1" font-family="Segoe UI, Inter, sans-serif" '
            f'font-size="12" font-weight="600">{metric_labels[m]}</text>'
        )
        lx += 165

    chart_left = 250
    chart_right = 805
    chart_width = chart_right - chart_left
    top_y = 122
    group_height = 92

    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        gx = chart_left + tick * chart_width
        svg.append(
            f'  <line x1="{gx:.1f}" y1="{top_y - 8}" x2="{gx:.1f}" y2="{top_y + 3 * group_height - 10}" '
            f'stroke="#334155" stroke-dasharray="4,4" stroke-width="1"/>'
        )
        svg.append(
            f'  <text x="{gx:.1f}" y="{top_y + 3 * group_height + 10}" fill="#64748b" '
            f'font-family="Segoe UI, Inter, sans-serif" font-size="11" text-anchor="middle">{tick:.2f}</text>'
        )

    for idx, entry in enumerate(results):
        gy = top_y + idx * group_height
        model_name = entry["model"]
        svg.append(
            f'  <text x="{chart_left - 14}" y="{gy + 36}" fill="#f1f5f9" '
            f'font-family="Segoe UI, Inter, sans-serif" font-size="13" font-weight="600" text-anchor="end">{model_name}</text>'
        )
        for m_idx, m in enumerate(metrics):
            val = entry[m]
            bar_w = max(2.0, val * chart_width)
            by = gy + m_idx * 18
            svg.append(
                f'  <rect x="{chart_left}" y="{by}" width="{bar_w:.1f}" height="14" rx="3" fill="{colors[m]}" opacity="0.92"/>'
            )
            svg.append(
                f'  <text x="{chart_left + bar_w + 8:.1f}" y="{by + 11}" fill="#e2e8f0" '
                f'font-family="Consolas, monospace" font-size="11" font-weight="600">{val:.4f}</text>'
            )

    svg.append('</svg>')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(svg), encoding="utf-8")


def main():
    X_text, y = generate_adversarial_news_corpus()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "TF-IDF + Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=3000)),
            ("clf", LogisticRegression(C=1.0, max_iter=500, random_state=42)),
        ]),
        "TF-IDF + Passive-Aggressive": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=3000)),
            ("clf", PassiveAggressiveClassifier(C=0.25, max_iter=1000, random_state=42)),
        ]),
        "TF-IDF + Complement Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=3000)),
            ("clf", ComplementNB(alpha=0.5)),
        ]),
    }

    benchmark_rows = []
    for name, model in models.items():
        oof_scores = np.zeros(len(y))
        oof_preds = np.zeros(len(y))
        for train_idx, val_idx in cv.split(X_text, y):
            model.fit(X_text[train_idx], y[train_idx])
            preds = model.predict(X_text[val_idx])
            oof_preds[val_idx] = preds
            if hasattr(model, "predict_proba"):
                oof_scores[val_idx] = model.predict_proba(X_text[val_idx])[:, 1]
            else:
                oof_scores[val_idx] = model.decision_function(X_text[val_idx])

        row = {
            "model": name,
            "accuracy": round(float(accuracy_score(y, oof_preds)), 4),
            "precision": round(float(precision_score(y, oof_preds)), 4),
            "recall": round(float(recall_score(y, oof_preds)), 4),
            "f1": round(float(f1_score(y, oof_preds)), 4),
            "roc_auc": round(float(roc_auc_score(y, oof_scores)), 4),
        }
        benchmark_rows.append(row)

    benchmark_rows.sort(key=lambda r: r["f1"], reverse=True)

    payload = {
        "project": "Adversarial Fake News & Misinformation NLP Classifier",
        "dataset": "Synthesized Adversarial Disinformation Corpus (N=1,200, 1-2 n-grams)",
        "validation_scheme": "5-Fold Stratified Cross-Validation",
        "results": benchmark_rows,
    }

    base_dir = Path(__file__).resolve().parent
    json_path = base_dir / "benchmarks" / "benchmark_results.json"
    svg_path = base_dir / "assets" / "benchmark_chart.svg"

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    render_svg_chart(benchmark_rows, svg_path)
    print(f"Saved benchmark results to {json_path}")
    print(f"Saved benchmark chart to {svg_path}")


if __name__ == "__main__":
    main()
