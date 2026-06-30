"""
evaluate_classifier.py — Full evaluation: confusion matrix + per-category
precision/recall, comparing keyword-only classification against the
keyword + TF-IDF/LogReg fallback pipeline.

This is the actual receipt behind every accuracy claim in the app and README.
Run directly to reproduce:

    python3 evaluate_classifier.py

Evaluation is done on test_labels.csv — held-out data the ML fallback model
never saw during training (train_labels.csv). The keyword matcher has no
"training" step (it's a static rule set) so it's evaluated on the same held-out
set for a fair comparison.
"""
import csv
from collections import defaultdict

from classifier import infer_category
from ml_classifier import load_model, categorize_with_fallback


def load_test_set(path="test_labels.csv"):
    with open(path) as f:
        reader = csv.DictReader(f)
        return [(r["description"], r["true_category"]) for r in reader]


def confusion_matrix(predictions, categories):
    """predictions: list of (true, predicted) tuples"""
    matrix = defaultdict(lambda: defaultdict(int))
    for true_cat, pred_cat in predictions:
        matrix[true_cat][pred_cat] += 1
    return matrix


def precision_recall_f1(predictions, categories):
    """Returns dict: category -> {precision, recall, f1, support}"""
    stats = {}
    for cat in categories:
        tp = sum(1 for t, p in predictions if t == cat and p == cat)
        fp = sum(1 for t, p in predictions if t != cat and p == cat)
        fn = sum(1 for t, p in predictions if t == cat and p != cat)
        support = sum(1 for t, p in predictions if t == cat)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0 else 0.0)

        stats[cat] = {
            "precision": precision, "recall": recall,
            "f1": f1, "support": support,
        }
    return stats


def print_confusion_matrix(matrix, categories):
    short = {c: c[:8] for c in categories}
    header = "TRUE\\PRED".ljust(20) + "".join(short[c].rjust(10) for c in categories)
    print(header)
    print("-" * len(header))
    for true_cat in categories:
        row = true_cat.ljust(20)
        for pred_cat in categories:
            row += str(matrix[true_cat].get(pred_cat, 0)).rjust(10)
        print(row)


def print_precision_recall(stats, categories, label):
    print(f"\n{label} — per-category precision / recall / f1")
    print("-" * 64)
    print(f"{'Category':<20}{'Precision':>11}{'Recall':>11}{'F1':>9}{'Support':>10}")
    for cat in categories:
        s = stats[cat]
        print(f"{cat:<20}{s['precision']*100:>10.1f}%{s['recall']*100:>10.1f}%"
              f"{s['f1']:>9.2f}{s['support']:>10}")

    macro_p = sum(s["precision"] for s in stats.values()) / len(stats)
    macro_r = sum(s["recall"] for s in stats.values()) / len(stats)
    macro_f1 = sum(s["f1"] for s in stats.values()) / len(stats)
    print("-" * 64)
    print(f"{'MACRO AVG':<20}{macro_p*100:>10.1f}%{macro_r*100:>10.1f}%{macro_f1:>9.2f}")


def main():
    test_set = load_test_set()
    categories = sorted(set(t for _, t in test_set))

    print("=" * 64)
    print(f"CLASSIFIER EVALUATION — {len(test_set)} held-out test transactions")
    print("(test_labels.csv — never seen during ML fallback training)")
    print("=" * 64)

    # ── Pipeline 1: keyword-only ────────────────────────────────────────────
    keyword_preds = [(true, infer_category(desc)) for desc, true in test_set]
    keyword_correct = sum(1 for t, p in keyword_preds if t == p)
    keyword_acc = keyword_correct / len(test_set) * 100

    # ── Pipeline 2: keyword + ML fallback ───────────────────────────────────
    model = load_model()
    fallback_preds = []
    method_counts = defaultdict(int)
    for desc, true in test_set:
        pred, method = categorize_with_fallback(desc, model)
        fallback_preds.append((true, pred))
        method_counts[method] += 1
    fallback_correct = sum(1 for t, p in fallback_preds if t == p)
    fallback_acc = fallback_correct / len(test_set) * 100

    print(f"\nOVERALL ACCURACY")
    print("-" * 64)
    print(f"  Keyword-only:           {keyword_correct}/{len(test_set)} = {keyword_acc:.1f}%")
    print(f"  Keyword + ML fallback:  {fallback_correct}/{len(test_set)} = {fallback_acc:.1f}%")
    print(f"  Improvement:            {fallback_acc - keyword_acc:+.1f} percentage points")

    print(f"\nRESOLUTION METHOD BREAKDOWN (keyword + ML fallback pipeline)")
    print("-" * 64)
    for method, count in sorted(method_counts.items()):
        pct = count / len(test_set) * 100
        print(f"  {method:<15} {count:>3} transactions ({pct:.1f}%)")

    print(f"\n{'='*64}")
    print("CONFUSION MATRIX — Keyword-only")
    print("="*64)
    cm_keyword = confusion_matrix(keyword_preds, categories)
    print_confusion_matrix(cm_keyword, categories)

    print(f"\n{'='*64}")
    print("CONFUSION MATRIX — Keyword + ML fallback")
    print("="*64)
    cm_fallback = confusion_matrix(fallback_preds, categories)
    print_confusion_matrix(cm_fallback, categories)

    stats_keyword = precision_recall_f1(keyword_preds, categories)
    stats_fallback = precision_recall_f1(fallback_preds, categories)

    print(f"\n{'='*64}")
    print_precision_recall(stats_keyword, categories, "KEYWORD-ONLY")
    print(f"\n{'='*64}")
    print_precision_recall(stats_fallback, categories, "KEYWORD + ML FALLBACK")

    print(f"\n{'='*64}")
    print("MISCLASSIFICATIONS — Keyword + ML fallback (remaining errors)")
    print("="*64)
    errors = [(d, t, p) for (d, t), (_, p) in zip(test_set, fallback_preds) if t != p]
    if errors:
        for desc, true, pred in errors:
            print(f"  '{desc}'")
            print(f"      true: {true:<18} predicted: {pred}")
    else:
        print("  None — perfect classification on held-out test set.")

    print(f"\n{'='*64}")
    print(f"RESULT: keyword-only {keyword_acc:.1f}% -> keyword+ML fallback {fallback_acc:.1f}%")
    print(f"on {len(test_set)} held-out transactions (trained on separate {175}-row set)")
    print("="*64)

    return fallback_acc


if __name__ == "__main__":
    main()
