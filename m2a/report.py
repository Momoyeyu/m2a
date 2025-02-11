"""
Aggregate experiment-cell results into the paper-style 2-D tables.

Each cell of a table corresponds to one (method, model) pair and reports
``[EP, ASR, UER, SNR]`` — the mean of the per-attack-type aggregates, i.e.
``(mirage_result + mute_result) / 2``, matching the protocol in the paper.
"""

import glob
import json
import os

METRICS = ("EP", "ASR", "UER", "SNR")


def collect_results(results_dir):
    """Load every ``*.json`` written by ``runner.run_experiment``."""
    results = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path) as f:
            data = json.load(f)
        key = (data["model"], data["method"], data["attack_type"])
        results[key] = data
    return results


def cell_metrics(results, model, method, attack_types):
    """Mean of the per-attack-type aggregates for one (model, method)."""
    values = []
    for at in attack_types:
        entry = results.get((model, method, at))
        if entry is None or entry.get("aggregate") is None:
            continue
        values.append([entry["aggregate"][m] for m in METRICS])
    if not values:
        return None
    return [sum(col) / len(col) for col in zip(*values)]


def build_table(results, methods, models, attack_types=("mirage", "mute")):
    """
    Returns a 2-D python list whose dimensions match the paper tables:
    rows = attack methods, columns = models x [EP, ASR, UER, SNR].
    """
    table = []
    for method in methods:
        row = []
        for model in models:
            metrics = cell_metrics(results, model, method, attack_types)
            row.extend(metrics if metrics is not None else [None] * len(METRICS))
        table.append(row)
    return table


def format_pct(value):
    return "None" if value is None else f"{value * 100:.2f}%"


def format_db(value):
    return "None" if value is None else f"{value:.2f}dB"


def print_table(table, methods, models, title=None):
    """Pretty-print the table, then print the raw 2-D python array."""
    if title:
        print(f"\n=== {title} ===")
    header = ["Method"] + [f"{m}" for m in models for _ in METRICS]
    print("".join(f"{h:<12}" for h in header))
    print(" " * 12 + "".join(f"{metric:<12}" for _ in models for metric in METRICS))
    for method, row in zip(methods, table):
        cells = []
        for j, v in enumerate(row):
            cells.append(format_db(v) if j % len(METRICS) == 3 else format_pct(v))
        print(f"{method:<12}" + "".join(f"{c:<12}" for c in cells))
    print("\n2D array (rows=methods, cols=model x [EP, ASR, UER, SNR]):")
    print(table)
    return table


def save_table(table, methods, models, path, title=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "title": title,
        "methods": methods,
        "models": models,
        "metrics": list(METRICS),
        "table": table,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Table saved to {path}")
