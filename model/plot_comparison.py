"""SVD ve NMF'nin RMSE/MAE sonuclarini karsilastiran basit bir cubuk grafik uretir."""
import matplotlib.pyplot as plt
import numpy as np

# rapor_svd.txt / rapor_nmf.txt'deki sonuclar
svd = {"RMSE": 0.3071, "MAE": 0.2338}
nmf = {"RMSE": 0.3258, "MAE": 0.2308}

metrics = list(svd.keys())
x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)

bars_svd = ax.bar(x - width / 2, [svd[m] for m in metrics], width, label="SVD", color="#2a78d6")
bars_nmf = ax.bar(x + width / 2, [nmf[m] for m in metrics], width, label="NMF", color="#eb6834")

# Her cubugun tepesine degerini yaz
for bars in (bars_svd, bars_nmf):
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)

ax.set_ylabel("Hata degeri")
ax.set_title("SVD vs NMF - RMSE ve MAE Karsilastirmasi")
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 0.4)
ax.legend()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle="-", alpha=0.2)

fig.tight_layout()
fig.savefig("model/svd_vs_nmf_matplotlib.png")
print("Kaydedildi: model/svd_vs_nmf_matplotlib.png")
