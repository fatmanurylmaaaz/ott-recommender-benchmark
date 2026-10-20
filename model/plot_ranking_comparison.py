"""Ranking degerlendirmesindeki (Precision@10 / Recall@10) sonuclari,
SVD/NMF'yi rastgele ve populerlik baseline'lariyla karsilastiran bir
grafik olarak kaydeder. rapor_ranking.txt'deki degerlerle ayni."""
import matplotlib.pyplot as plt
import numpy as np

# model/rapor_ranking.txt'deki degerler (K=10)
methods = ["SVD", "NMF", "Rastgele", "En Populer"]
precision = [0.00004, 0.00037, 0.00064, 0.04635]
recall = [0.0004, 0.0037, 0.0064, 0.4635]
colors = ["#2a78d6", "#eb6834", "#898781", "#0ca30c"]

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=150)

for ax, values, title in zip(axes, [precision, recall], ["Precision@10", "Recall@10"]):
    bars = ax.bar(methods, values, color=colors)
    ax.set_yscale("log")  # deger araligi 4 buyukluk mertebesi farkli oldugu icin log olcek sart
    ax.set_title(title)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", which="both", linestyle="-", alpha=0.15)
    ax.tick_params(axis="x", rotation=15)

fig.suptitle("SVD/NMF vs Rastgele/En Populer Baseline (K=10, log olcek)", fontsize=12)
fig.tight_layout()
fig.savefig("model/ranking_comparison.png")
print("Kaydedildi: model/ranking_comparison.png")
