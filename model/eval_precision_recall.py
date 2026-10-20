"""SVD ve NMF icin leave-one-out Precision@K / Recall@K (Hit Rate@K) hesaplar.

Yontem: her kullanici icin train'de gormedigi tum icerikleri modelin ogrendigi
gizli faktorlerle (pu, qi, bias) puanlayip siraliyoruz, top-K listesinde
test.csv'deki gercek (tutulan) icerik var mi diye bakiyoruz. Bu, Surprise'in
algo.predict() metodunu milyonlarca kez cagirmak yerine numpy ile toplu
(vektorize) matris carpimi kullaniyor - cok daha hizli.
"""
import pickle
import sys

import numpy as np
import pandas as pd

from common import load_config, load_trainset


def load_test_relevant(test_path, trainset):
    """user_id -> test.csv'deki gercek icerik(ler)in inner id kumesi."""
    df = pd.read_csv(test_path)
    relevant = {}
    unknown_items = 0
    for row in df.itertuples(index=False):
        try:
            inner_uid = trainset.to_inner_uid(str(row.user_id))
        except ValueError:
            continue  # trainset'te olmayan kullanici (beklenmiyor)
        try:
            # content_id, train.csv okunurken int64 olarak yukleniyor (pandas
            # otomatik cikarim yapiyor) - trainset'teki raw id'ler de ayni tipte
            # saklaniyor, o yuzden str()'e cevirmiyoruz (aksi halde hicbiri eslesmez).
            inner_iid = trainset.to_inner_iid(row.content_id)
        except ValueError:
            unknown_items += 1
            continue  # trainset'te hic gorulmemis icerik (cold item)
        relevant.setdefault(inner_uid, set()).add(inner_iid)
    return relevant, unknown_items


def score_matrix(algo, pu_batch):
    """(kullanici_sayisi, n_factors) -> (kullanici_sayisi, n_items) skor matrisi."""
    scores = pu_batch @ algo.qi.T
    if algo.biased:
        scores += algo.trainset.global_mean
        scores += algo.bi[np.newaxis, :]
    return scores


def evaluate(algo, trainset, relevant, k, batch_size):
    inner_uids = sorted(relevant.keys())
    n_items = trainset.n_items
    hits = 0
    precision_sum = 0.0
    recall_sum = 0.0
    n_users = len(inner_uids)

    for start in range(0, n_users, batch_size):
        batch_uids = inner_uids[start:start + batch_size]
        pu_batch = algo.pu[batch_uids]

        scores = score_matrix(algo, pu_batch)
        if algo.biased:
            bu_batch = algo.bu[batch_uids]
            scores += bu_batch[:, np.newaxis]

        # Train'de zaten gorulmus icerikleri diskla - onlari tekrar onermeyelim.
        for row_idx, inner_uid in enumerate(batch_uids):
            seen = [iid for iid, _ in trainset.ur[inner_uid]]
            scores[row_idx, seen] = -np.inf

        # Her satir icin en yuksek skorlu K indeksi bul (siralamaya gerek yok,
        # sadece "top-K kumesinde mi" sorusu icin argpartition yeterli ve hizli).
        top_k_idx = np.argpartition(-scores, kth=min(k, n_items - 1), axis=1)[:, :k]

        for row_idx, inner_uid in enumerate(batch_uids):
            true_items = relevant[inner_uid]
            hit_count = len(true_items.intersection(top_k_idx[row_idx]))
            if hit_count > 0:
                hits += 1
            precision_sum += hit_count / k
            recall_sum += hit_count / len(true_items)

    return {
        "n_users": n_users,
        "hit_rate": hits / n_users,
        "precision_at_k": precision_sum / n_users,
        "recall_at_k": recall_sum / n_users,
    }


def evaluate_random(trainset, relevant, k, seed=42):
    """Baseline: her kullaniciya rastgele K icerik oner. Modelin gercekten
    bir sey ogrenip ogrenmedigini anlamak icin taban referans."""
    rng = np.random.default_rng(seed)
    n_items = trainset.n_items
    n_users = len(relevant)
    hits = 0
    precision_sum = 0.0
    recall_sum = 0.0
    for uid, true_items in relevant.items():
        seen = set(iid for iid, _ in trainset.ur[uid])
        candidates = np.array([i for i in range(n_items) if i not in seen])
        picks = rng.choice(candidates, size=min(k, len(candidates)), replace=False)
        hit_count = len(true_items.intersection(picks))
        if hit_count > 0:
            hits += 1
        precision_sum += hit_count / k
        recall_sum += hit_count / len(true_items)
    return {
        "n_users": n_users,
        "hit_rate": hits / n_users,
        "precision_at_k": precision_sum / n_users,
        "recall_at_k": recall_sum / n_users,
    }


def evaluate_most_popular(trainset, relevant, k):
    """Baseline: herkese ayni sekilde train'de en cok etkilesim alan K
    icerigi oner (kisisellestirme yok). Katalog kucuk/carpik oldugu icin
    bu, sasirtici derecede guclu bir referans cikabilir."""
    n_items = trainset.n_items
    n_users = len(relevant)
    item_counts = np.zeros(n_items, dtype=int)
    for uid in range(trainset.n_users):
        for iid, _ in trainset.ur[uid]:
            item_counts[iid] += 1
    most_popular_order = np.argsort(-item_counts)

    hits = 0
    precision_sum = 0.0
    recall_sum = 0.0
    for uid, true_items in relevant.items():
        seen = set(iid for iid, _ in trainset.ur[uid])
        top_k = [i for i in most_popular_order if i not in seen][:k]
        hit_count = len(true_items.intersection(top_k))
        if hit_count > 0:
            hits += 1
        precision_sum += hit_count / k
        recall_sum += hit_count / len(true_items)
    return {
        "n_users": n_users,
        "hit_rate": hits / n_users,
        "precision_at_k": precision_sum / n_users,
        "recall_at_k": recall_sum / n_users,
    }


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "model/eval_config.yaml"
    config = load_config(config_path)
    k = config["k"]

    print("Trainset yukleniyor (pu/qi siralamasi model ile eslessin diye ayni sekilde kuruluyor)...")
    trainset = load_trainset(config["train_path"], config["rating_scale"])
    relevant, unknown_items = load_test_relevant(config["test_path"], trainset)
    print(f"Degerlendirilecek kullanici sayisi: {len(relevant)}")
    print(f"Trainset'te bulunamayan (cold) test icerik sayisi: {unknown_items}")
    print()

    results = {}
    for name, model_path in [("SVD", config["svd_model_path"]), ("NMF", config["nmf_model_path"])]:
        print(f"{name} degerlendiriliyor (K={k})...")
        with open(model_path, "rb") as f:
            algo = pickle.load(f)
        res = evaluate(algo, trainset, relevant, k, config["batch_size"])
        results[name] = res
        print(
            f"  {name} -> HitRate@{k}: {res['hit_rate']:.4f}  "
            f"Precision@{k}: {res['precision_at_k']:.4f}  "
            f"Recall@{k}: {res['recall_at_k']:.4f}"
        )

    # Baseline'lar: SVD/NMF'nin gercekten "bir sey ogrenip ogrenmedigini"
    # anlamak icin - hem rastgele oneri hem de kisisellestirme yapmayan
    # "en populer K icerik" ile karsilastiriyoruz.
    print("Rastgele oneri (baseline) degerlendiriliyor...")
    results["Rastgele"] = evaluate_random(trainset, relevant, k)
    print(f"  Rastgele -> HitRate@{k}: {results['Rastgele']['hit_rate']:.4f}")

    print("En populer icerik (baseline) degerlendiriliyor...")
    results["En Populer"] = evaluate_most_popular(trainset, relevant, k)
    print(f"  En Populer -> HitRate@{k}: {results['En Populer']['hit_rate']:.4f}")

    lines = [
        "RANKING DEGERLENDIRMESI (leave-one-out Precision/Recall/HitRate)",
        "=" * 65,
        "",
        f"K: {k}",
        f"Degerlendirilen kullanici sayisi: {len(relevant)}",
        f"Trainset'te bulunamayan (cold) test icerik sayisi: {unknown_items}",
        "",
        f"{'Model':<8}{'HitRate@K':>12}{'Precision@K':>14}{'Recall@K':>12}",
    ]
    for name, res in results.items():
        lines.append(
            f"{name:<8}{res['hit_rate']:>12.4f}{res['precision_at_k']:>14.4f}{res['recall_at_k']:>12.4f}"
        )
    lines.append("")
    lines.append(
        "Not: Her kullanicinin test setinde 1 (nadiren 2) ilgili icerigi oldugu icin "
        "Recall@K burada pratikte HitRate@K'ya esdegerdir; Precision@K ise K=10 "
        "onerinin ne kadarinin isabetli oldugunu gosterir."
    )
    lines.append("")
    lines.append("YORUM")
    lines.append("-" * 65)
    lines.append(
        "SVD ve NMF, 'En Populer' baseline'inin (kisisellestirme yapmayan, "
        "herkese ayni top-K listeyi oneren) COK gerisinde kaldi - hatta SVD "
        "rastgele oneriden bile kotu cikti. Sebep muhtemelen su: SVD/NMF "
        "'rating' (izlenen/toplam sure orani) degerini dogru tahmin etmeye "
        "(RMSE/MAE) optimize edildi - bu, 'kullanici bu icerigi izlerse ne "
        "kadar tamamlar' sorusuna cevap veriyor, 'kullanici hangi icerigi "
        "izleyecek' sorusuna degil. Katalog kucuk (1.535 icerik) ve izlenme "
        "dagilimi asiri carpik oldugu icin (bkz. trending/ klasoru), sadece "
        "populerlige bakmak bile cok guclu bir sinyal - ama SVD/NMF bu "
        "sinyali ogrenmiyor, cunku egitim hedefleri hic bu degildi."
    )

    with open(config["report_path"], "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print()
    print(f"Rapor kaydedildi: {config['report_path']}")


if __name__ == "__main__":
    main()
