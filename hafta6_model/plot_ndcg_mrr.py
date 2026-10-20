"""5 yontemin (SVD, NMF, BPR, Rastgele, En Populer) NDCG@10 ve MRR'sini
topk_lists.pkl'deki SIRALI listelerden hesaplayip grafik + rapor uretir.

NDCG@10: isabetin kacinci sirada oldugunu logaritmik agirlikla odullendirir,
         ideal siralamaya (IDCG) oranlanir (0-1 arasi).
MRR:     ilk isabetin 1/sira degeri, kullanicilar uzerinden ortalanir.
"""
import math
import pickle

import matplotlib.pyplot as plt

K = 10

with open("hafta6_model/data/topk_lists.pkl", "rb") as f:
    data = pickle.load(f)

relevant = data["relevant"]
topk = data["topk"]

order = ["SVD", "NMF", "Rastgele", "BPR", "En Populer"]
colors = ["#2a78d6", "#eb6834", "#898781", "#7a3fb5", "#0ca30c"]


def ndcg_at_k(items, true_items, k):
    dcg = 0.0
    for i, item in enumerate(items[:k], start=1):
        if item in true_items:
            dcg += 1.0 / math.log2(i + 1)
    # Ideal siralama: tum ilgili icerikler (en fazla k tanesi) en basta olsaydi.
    ideal_hits = min(len(true_items), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def mrr_at_k(items, true_items, k):
    for i, item in enumerate(items[:k], start=1):
        if item in true_items:
            return 1.0 / i
    return 0.0


ndcg_scores = []
mrr_scores = []
for method in order:
    lists = topk[method]
    ndcg_sum = 0.0
    mrr_sum = 0.0
    n = 0
    for uid, items in lists.items():
        true_items = relevant.get(uid)
        if not true_items:
            continue
        ndcg_sum += ndcg_at_k(items, true_items, K)
        mrr_sum += mrr_at_k(items, true_items, K)
        n += 1
    ndcg_scores.append(ndcg_sum / n)
    mrr_scores.append(mrr_sum / n)

lines = ["NDCG@10 / MRR - 5 YONTEM", "=" * 40, ""]
lines.append(f"{'Method':<12}{'NDCG@10':>12}{'MRR':>12}")
for m, nd, mr in zip(order, ndcg_scores, mrr_scores):
    lines.append(f"{m:<12}{nd:>12.4f}{mr:>12.4f}")
with open("hafta6_model/reports/rapor_ndcg_mrr.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\n".join(lines))

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=150)
for ax, values, title in zip(axes, [ndcg_scores, mrr_scores], ["NDCG@10", "MRR"]):
    bars = ax.bar(order, values, color=colors)
    ax.set_yscale("log")
    ax.set_title(title)
    ax.bar_label(bars, labels=[f"{v:.4f}" for v in values], padding=3, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", which="both", linestyle="-", alpha=0.15)
    ax.tick_params(axis="x", rotation=15)

fig.suptitle("5 Yöntem: NDCG@10 ve MRR (log ölçek)", fontsize=12)
fig.tight_layout()
fig.savefig("hafta6_model/charts/ndcg_mrr_comparison.png")
print("\nKaydedildi: hafta6_model/charts/ndcg_mrr_comparison.png")
print("Kaydedildi: hafta6_model/reports/rapor_ndcg_mrr.txt")
