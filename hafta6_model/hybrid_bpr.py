"""BPR skorunu populerlik skoruyla harmanlayip (hibrit) birkac alpha
degerini HitRate@10 (hizli proxy metrik) uzerinden karsilastirir.

final_score = alpha * BPR_score_normalized + (1 - alpha) * populerlik_score_normalized

- alpha = 1.0  -> saf BPR (hibrit yok)
- alpha = 0.0  -> saf populerlik (En Populer baseline'iyla ayni davranir)
- Aradaki degerler iki sinyali karistirir.

BPR skorlarini implicit'in kendi model.recommend() metoduyla aliyoruz
(N=n_items, filter_already_liked_items=True) - boylece bias/seen-item
filtreleme mantigini yeniden yazmak yerine kutuphanenin dogru sonucuna
guveniyoruz. Donen skorlar kullanici bazinda min-max normalize edilip
populerlik skoruyla harmanlaniyor.
"""
import pickle
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

from eval_common import load_test_relevant

BATCH_SIZE = 20000
ALPHAS = [0.3, 0.5, 0.7]


def minmax_norm(arr, axis=None):
    """Satir bazinda (axis=1) ya da global (axis=None) min-max normalizasyon.
    Sabit satirlarda (max==min) sifira bolme olmasin diye kucuk bir epsilon eklenir."""
    a_min = arr.min(axis=axis, keepdims=(axis is not None))
    a_max = arr.max(axis=axis, keepdims=(axis is not None))
    return (arr - a_min) / (a_max - a_min + 1e-12)


def main():
    print("Model ve veriler yukleniyor...")
    with open("hafta6_model/data/bpr_model.pkl", "rb") as f:
        model = pickle.load(f)
    train_matrix = sp.load_npz("hafta6_model/data/train_matrix.npz").tocsr()
    user_map = pd.read_csv("hafta6_model/data/user_index_map.csv")
    item_map = pd.read_csv("hafta6_model/data/item_index_map.csv").sort_values("item_idx")
    idx_to_item = item_map["content_id"].to_numpy()
    n_items = train_matrix.shape[1]

    relevant = load_test_relevant("model/test.csv")
    user_to_idx = dict(zip(user_map["user_id"].astype(str), user_map["user_idx"]))
    valid_users = [u for u in relevant if u in user_to_idx]
    user_indices = np.array([user_to_idx[u] for u in valid_users])
    print(f"Degerlendirilecek kullanici sayisi: {len(valid_users)}")

    # Populerlik skoru: train matrisindeki her icerigin toplam etkilesim sayisi,
    # global (tum kataloga gore) min-max normalize edilir.
    pop_counts = np.asarray(train_matrix.sum(axis=0)).ravel()
    pop_norm = minmax_norm(pop_counts)

    hits = {a: 0 for a in ALPHAS}
    t0 = time.time()

    for start in range(0, len(user_indices), BATCH_SIZE):
        batch_idx = user_indices[start:start + BATCH_SIZE]
        batch_real = valid_users[start:start + BATCH_SIZE]

        # implicit'in kendi recommend() metodu: bias'i dogru hesaplar, gorulmus
        # icerikleri otomatik eler. N=n_items ile kullanicinin gormedigi TUM
        # icerikleri skoruyla birlikte alip kendimiz yeniden siraliyoruz.
        ids, scores = model.recommend(
            batch_idx, train_matrix[batch_idx], N=n_items, filter_already_liked_items=True
        )

        bpr_norm = minmax_norm(scores, axis=1)
        item_pop = pop_norm[ids]  # (batch, n_items) - her satir kendi item sirasina gore

        for alpha in ALPHAS:
            final_scores = alpha * bpr_norm + (1 - alpha) * item_pop
            order = np.argsort(-final_scores, axis=1)[:, :10]
            for row_idx, real_uid in enumerate(batch_real):
                top_ids = ids[row_idx][order[row_idx]]
                top_content = set(int(idx_to_item[i]) for i in top_ids)
                if top_content.intersection(relevant[real_uid]):
                    hits[alpha] += 1

    elapsed = time.time() - t0
    n = len(valid_users)

    lines = ["HIBRIT (BPR + POPULERLIK) ALPHA DENEMESI", "=" * 50, ""]
    lines.append("final_score = alpha * BPR_norm + (1-alpha) * populerlik_norm")
    lines.append("")
    lines.append(f"{'Alpha':<10}{'HitRate@10':>14}")
    for a in ALPHAS:
        lines.append(f"{a:<10}{hits[a] / n:>14.4f}")
    lines.append("")
    lines.append(f"Referans - saf BPR (Config C):     0.2671")
    lines.append(f"Referans - En Populer baseline:      0.4635")
    lines.append("")
    best_alpha = max(ALPHAS, key=lambda a: hits[a])
    lines.append(f"En iyi alpha: {best_alpha} (HitRate@10={hits[best_alpha] / n:.4f})")
    lines.append(f"Toplam sure: {elapsed:.1f}sn ({n} kullanici)")

    with open("hafta6_model/reports/rapor_hybrid_tuning.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print("\nRapor kaydedildi: hafta6_model/reports/rapor_hybrid_tuning.txt")


if __name__ == "__main__":
    main()
