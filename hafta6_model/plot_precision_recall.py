"""5 yontemin (SVD, NMF, BPR, Rastgele, En Populer) Precision@10 ve
Recall@10'unu topk_lists.pkl'den hesaplayip grafik + rapor uretir."""
import pickle

import matplotlib.pyplot as plt

K = 10

with open("hafta6_model/data/topk_lists.pkl", "rb") as f:
    data = pickle.load(f)

relevant = data["relevant"]
topk = data["topk"]

order = ["SVD", "NMF", "Rastgele", "BPR", "En Populer"]
colors = ["#2a78d6", "#eb6834", "#898781", "#7a3fb5", "#0ca30c"]

precision = []
recall = []
for method in order:
    lists = topk[method]
    precision_sum = 0.0
    recall_sum = 0.0
    for uid, items in lists.items():
        true_items = relevant.get(uid)
        if not true_items:
            continue
        hit_count = len(true_items.intersection(items[:K]))
        precision_sum += hit_count / K
        recall_sum += hit_count / len(true_items)
    n = len(lists)
    precision.append(precision_sum / n)
    recall.append(recall_sum / n)

# Rapor
lines = ["PRECISION@10 / RECALL@10 - 5 YONTEM", "=" * 45, ""]
lines.append(f"{'Method':<12}{'Precision@10':>14}{'Recall@10':>12}")
for m, p, r in zip(order, precision, recall):
    lines.append(f"{m:<12}{p:>14.4f}{r:>12.4f}")
with open("hafta6_model/reports/rapor_precision_recall.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\n".join(lines))

# Grafik
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=150)
for ax, values, title in zip(axes, [precision, recall], ["Precision@10", "Recall@10"]):
    bars = ax.bar(order, values, color=colors)
    ax.set_yscale("log")
    ax.set_title(title)
    ax.bar_label(bars, labels=[f"{v:.4f}" for v in values], padding=3, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", which="both", linestyle="-", alpha=0.15)
    ax.tick_params(axis="x", rotation=15)

fig.suptitle("5 Yöntem: Precision@10 ve Recall@10 (log ölçek)", fontsize=12)
fig.tight_layout()
fig.savefig("hafta6_model/charts/precision_recall_comparison.png")
print("\nKaydedildi: hafta6_model/charts/precision_recall_comparison.png")
print("Kaydedildi: hafta6_model/reports/rapor_precision_recall.txt")
