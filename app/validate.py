"""
validate.py — Measures classifier accuracy against validation.csv

This is the actual receipt behind the "~89% accuracy" claim shown in the
app UI. Run this script directly to reproduce the number:

    python3 validate.py

It loads 100 manually labeled transactions (validation.csv), runs each
description through infer_category(), and reports overall accuracy plus
a per-category breakdown and a confusion list of every miss.

The validation set intentionally includes hard cases (abbreviated merchant
names, payment processor prefixes like SQ*/TST*/PAYPAL*, hyphenated names)
that the keyword matcher is known to miss — the accuracy number is not
cherry-picked from easy cases.
"""
import csv
from collections import defaultdict
from classifier import infer_category


def load_validation_set(path="validation.csv"):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row["description"], row["true_category"]))
    return rows


def run_validation(rows):
    correct = 0
    total   = len(rows)
    misses  = []
    per_category = defaultdict(lambda: {"correct": 0, "total": 0})

    for description, true_cat in rows:
        predicted = infer_category(description)
        per_category[true_cat]["total"] += 1
        if predicted == true_cat:
            correct += 1
            per_category[true_cat]["correct"] += 1
        else:
            misses.append((description, true_cat, predicted))

    return correct, total, per_category, misses


def main():
    rows = load_validation_set()
    correct, total, per_category, misses = run_validation(rows)
    accuracy = correct / total * 100

    print("=" * 64)
    print("CLASSIFIER VALIDATION — infer_category() vs. validation.csv")
    print("=" * 64)
    print(f"\nOverall accuracy: {correct}/{total} = {accuracy:.1f}%\n")

    print("Per-category breakdown:")
    print("-" * 64)
    for cat in sorted(per_category.keys()):
        c = per_category[cat]["correct"]
        t = per_category[cat]["total"]
        pct = (c / t * 100) if t > 0 else 0
        bar = "#" * int(pct / 5)
        print(f"  {cat:<20} {c:>3}/{t:<3}  {pct:5.1f}%  {bar}")

    print(f"\nMisclassifications ({len(misses)}):")
    print("-" * 64)
    for desc, true_cat, predicted in misses:
        print(f"  '{desc}'")
        print(f"      true: {true_cat:<18} predicted: {predicted}")

    print("\n" + "=" * 64)
    print(f"RESULT: {accuracy:.1f}% accuracy on {total} manually labeled transactions")
    print("=" * 64)

    return accuracy


if __name__ == "__main__":
    main()
