"""8 Metrik Degerlendirmesi - Adim C: Coverage, Diversity, Gini-Index.

Bu ucu, onceki metriklerden (Precision/Recall/HitRate/NDCG/MRR) farkli olarak
KULLANICI bazinda degil, SISTEM/KATALOG bazinda hesaplanir - yani "dogru
onerdik mi" degil, "onerilerimiz genel olarak ne kadar genis/cesitli/adil"
sorusuna bakar. topk_lists.pkl'deki hazir top-10 listelerinden hesaplanir,
yeniden model egitimi gerekmez.

Coverage:  Onerilen benzersiz icerik sayisi / toplam katalog buyuklugu.
           Sistem katalogun ne kadarini "kullanabiliyor".
Diversity: Her kullanicinin top-10 listesi icindeki tur (genre) cesitliligi
           (ortalama ikili Jaccard-farkliligi), kullanicilar uzerinden ortalanir.
Gini-Index: Onerilerin icerikler arasinda ne kadar esit dagildigi (0=tam esit,
           1=tamamen tekellesmis - hep ayni birkac icerik oneriliyor).
"""
import json
import pickle
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

GENRE_JSON = "intern_data (1)/all (1).json"
ITEM_MAP = "hafta6_model/data/item_index_map.csv"

order = ["SVD", "NMF", "Rastgele", "BPR", "En Populer"]
colors = ["#2a78d6", "#eb6834", "#898781", "#7a3fb5", "#0ca30c"]


def load_genre_map(json_path):
    """content_id (int) -> genre kumesi (set[str])."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    genre_map = {}
    for cid_str, meta in data.items():
        try:
            cid = int(cid_str)
        except ValueError:
            continue
        genre_map[cid] = set(meta.get("Genres") or [])
    return genre_map


def jaccard(a, b):
    """Iki tur kumesi arasindaki benzerlik. Ikisi de bossa (tur bilgisi yok)
    "ayni bilinmeyen kategori" kabul edilip benzerlik 1 (farklilik 0) dondurulur
    - eksik metadata cesitliligi yapay olarak sismesin diye."""
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def intra_list_diversity(items, genre_map):
    """Bir listedeki tum ikili kombinasyonlarin ortalama Jaccard-farkliligi."""
    pairs = list(combinations(items, 2))
    if not pairs:
        return None
    diffs = [1.0 - jaccard(genre_map.get(i, set()), genre_map.get(j, set())) for i, j in pairs]
    return sum(diffs) / len(diffs)


def gini_index(counts):
    """Siralanmis dagilim uzerinden standart Gini katsayisi formulu.
    counts: katalogdaki HER icerigin (0 dahil) kac kez onerildigini iceren dizi."""
    counts = np.sort(np.asarray(counts, dtype=float))
    n = len(counts)
    total = counts.sum()
    if total == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * counts)) / (n * total) - (n + 1) / n


def main():
    print("Veriler yukleniyor...")
    with open("hafta6_model/data/topk_lists.pkl", "rb") as f:
        data = pickle.load(f)
    topk = data["topk"]

    genre_map = load_genre_map(GENRE_JSON)
    catalog_ids = pd.read_csv(ITEM_MAP)["content_id"].tolist()
    catalog_size = len(catalog_ids)
    id_to_pos = {cid: i for i, cid in enumerate(catalog_ids)}
    print(f"Katalog buyuklugu: {catalog_size}")

    results = {}
    for method in order:
        lists = topk[method]

        # Coverage: onerilen tum listelerdeki benzersiz icerik sayisi / katalog
        recommended_union = set()
        item_counts = np.zeros(catalog_size, dtype=int)
        diversity_scores = []

        for items in lists.values():
            recommended_union.update(items)
            for cid in items:
                if cid in id_to_pos:
                    item_counts[id_to_pos[cid]] += 1
            d = intra_list_diversity(items, genre_map)
            if d is not None:
                diversity_scores.append(d)

        coverage = len(recommended_union) / catalog_size
        diversity = sum(diversity_scores) / len(diversity_scores)
        gini = gini_index(item_counts)

        results[method] = {"coverage": coverage, "diversity": diversity, "gini": gini}
        print(f"{method:<12} Coverage={coverage:.4f}  Diversity={diversity:.4f}  Gini={gini:.4f}")

    # Rapor
    lines = ["COVERAGE / DIVERSITY / GINI-INDEX - 5 YONTEM", "=" * 55, ""]
    lines.append(f"{'Method':<12}{'Coverage':>12}{'Diversity':>12}{'Gini':>12}")
    for m in order:
        r = results[m]
        lines.append(f"{m:<12}{r['coverage']:>12.4f}{r['diversity']:>12.4f}{r['gini']:>12.4f}")
    with open("hafta6_model/reports/rapor_coverage_diversity_gini.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n" + "\n".join(lines))

    # Grafik
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=150)
    metrics = ["coverage", "diversity", "gini"]
    titles = ["Coverage", "Diversity", "Gini-Index"]
    for ax, metric, title in zip(axes, metrics, titles):
        values = [results[m][metric] for m in order]
        bars = ax.bar(order, values, color=colors)
        ax.set_title(title)
        ax.bar_label(bars, labels=[f"{v:.3f}" for v in values], padding=3, fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="-", alpha=0.15)
        ax.tick_params(axis="x", rotation=15)
        ax.set_ylim(0, 1.05)

    fig.suptitle("5 Yöntem: Coverage / Diversity / Gini-Index", fontsize=12)
    fig.tight_layout()
    fig.savefig("hafta6_model/charts/coverage_diversity_gini.png")
    print("\nKaydedildi: hafta6_model/charts/coverage_diversity_gini.png")
    print("Kaydedildi: hafta6_model/reports/rapor_coverage_diversity_gini.txt")


if __name__ == "__main__":
    main()
