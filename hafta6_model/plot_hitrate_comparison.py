"""5 yontemin (SVD, NMF, BPR, Rastgele, En Populer) HitRate@10'unu
karsilastiran bar grafigi. topk_lists.pkl'den canli hesaplaniyor."""
import pickle

import matplotlib.pyplot as plt

with open("hafta6_model/data/topk_lists.pkl", "rb") as f:
    data = pickle.load(f)

relevant = data["relevant"]
topk = data["topk"]

# Sabit siralama: kotu -> iyi (grafikte soldan saga okunabilir bir hikaye olsun)
order = ["SVD", "NMF", "Rastgele", "BPR", "En Populer"]
colors = ["#2a78d6", "#eb6834", "#898781", "#7a3fb5", "#0ca30c"]

hit_rates = []
for method in order:
    lists = topk[method]
    hits = sum(1 for uid, items in lists.items() if uid in relevant and relevant[uid].intersection(items))
    hit_rates.append(hits / len(lists))

fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
bars = ax.bar(order, hit_rates, color=colors)
ax.set_yscale("log")
ax.set_ylabel("HitRate@10 (log ölçek)")
ax.set_title("5 Yöntemin HitRate@10 Karşılaştırması")
ax.bar_label(bars, labels=[f"{v:.4f}" for v in hit_rates], padding=3, fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", which="both", linestyle="-", alpha=0.15)

fig.tight_layout()
fig.savefig("hafta6_model/charts/hitrate_comparison.png")
print("Kaydedildi: hafta6_model/charts/hitrate_comparison.png")
for m, v in zip(order, hit_rates):
    print(f"  {m}: {v:.4f}")
