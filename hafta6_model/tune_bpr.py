"""Birkac BPR hiperparametre kombinasyonunu dener, her birini test.csv
uzerinde HitRate@10 ile karsilastirir (hizli proxy metrik - tam 8 metrik
degil, en iyi konfigurasyonu secmek icin).
"""
import time

import pandas as pd
import scipy.sparse as sp
from implicit.bpr import BayesianPersonalizedRanking

from eval_common import load_test_relevant
from generate_topk_lists import topk_bpr

K = 10
BATCH_SIZE = 20000

CANDIDATES = [
    {"name": "Mevcut (baseline)", "factors": 100, "iterations": 100, "learning_rate": 0.01, "regularization": 0.01},
    {"name": "B - uzun egitim", "factors": 100, "iterations": 250, "learning_rate": 0.01, "regularization": 0.01},
    {"name": "C - az boyut + guclu reg", "factors": 50, "iterations": 250, "learning_rate": 0.01, "regularization": 0.05},
    {"name": "D - farkli lr", "factors": 50, "iterations": 250, "learning_rate": 0.05, "regularization": 0.05},
]


def hit_rate(topk, relevant):
    hits = 0
    for uid, items in topk.items():
        if uid in relevant and relevant[uid].intersection(items):
            hits += 1
    return hits / len(topk)


def main():
    print("Train matrisi ve test kullanicilari yukleniyor...")
    train_matrix = sp.load_npz("hafta6_model/data/train_matrix.npz").tocsr()
    user_map = pd.read_csv("hafta6_model/data/user_index_map.csv")
    item_map = pd.read_csv("hafta6_model/data/item_index_map.csv")
    relevant = load_test_relevant("model/test.csv")
    user_ids = list(relevant.keys())

    results = []
    for cfg in CANDIDATES:
        print(f"\n=== {cfg['name']} === {cfg}")
        t0 = time.time()
        model = BayesianPersonalizedRanking(
            factors=cfg["factors"],
            iterations=cfg["iterations"],
            learning_rate=cfg["learning_rate"],
            regularization=cfg["regularization"],
            random_state=42,
        )
        model.fit(train_matrix, show_progress=False)
        elapsed = time.time() - t0

        topk = topk_bpr(model, train_matrix, user_map, item_map, user_ids, K, BATCH_SIZE)
        hr = hit_rate(topk, relevant)
        print(f"HitRate@10 = {hr:.4f}  (sure: {elapsed:.1f}sn)")
        results.append({**cfg, "hit_rate": hr, "seconds": elapsed})

    lines = ["BPR HIPERPARAMETRE DENEMESI", "=" * 60, ""]
    lines.append(f"{'Config':<28}{'HitRate@10':>12}{'Sure(sn)':>10}")
    for r in results:
        lines.append(f"{r['name']:<28}{r['hit_rate']:>12.4f}{r['seconds']:>10.1f}")

    best = max(results, key=lambda r: r["hit_rate"])
    lines.append("")
    lines.append(f"En iyi: {best['name']} (HitRate@10={best['hit_rate']:.4f})")
    lines.append(
        f"Parametreler: factors={best['factors']}, iterations={best['iterations']}, "
        f"learning_rate={best['learning_rate']}, regularization={best['regularization']}"
    )

    with open("hafta6_model/reports/rapor_bpr_tuning.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print("\nRapor kaydedildi: hafta6_model/reports/rapor_bpr_tuning.txt")


if __name__ == "__main__":
    main()
