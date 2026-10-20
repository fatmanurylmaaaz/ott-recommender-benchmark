"""5 yontemin (SVD, NMF, BPR, Rastgele, En Populer) her test kullanicisi
icin SIRALI top-10 oneri listesini uretir ve tek bir dosyaya kaydeder.

Onceki eval_precision_recall.py argpartition (sirasiz) kullaniyordu - bu
yeterliydi cunku sadece "top-K'da mi" sorusuna bakiyorduk. NDCG ve MRR ise
"kacinci sirada" bilgisine ihtiyac duyuyor, bu yuzden burada tam siralama
(argsort / implicit'in kendi recommend() metodu) kullaniliyor.

Cikti: hafta6_model/data/topk_lists.pkl
    { method_adi: { user_id(str): [content_id(int), ...] (siral, uzunluk<=K) } }
"""
import pickle
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, "model")
from common import load_config, load_trainset  # noqa: E402

from eval_common import load_test_relevant  # noqa: E402

K = 10
BATCH_SIZE = 20000


def topk_svd_nmf(algo, trainset, user_ids, k, batch_size):
    valid_users = [u for u in user_ids if u in trainset._raw2inner_id_users]
    inner_uids = np.array([trainset.to_inner_uid(u) for u in valid_users])
    n_items = trainset.n_items
    result = {}

    for start in range(0, len(inner_uids), batch_size):
        batch_inner = inner_uids[start:start + batch_size]
        batch_real = valid_users[start:start + batch_size]

        scores = algo.pu[batch_inner] @ algo.qi.T
        if algo.biased:
            scores += algo.trainset.global_mean
            scores += algo.bi[np.newaxis, :]
            scores += algo.bu[batch_inner][:, np.newaxis]

        for row_idx, inner_uid in enumerate(batch_inner):
            seen = [iid for iid, _ in trainset.ur[inner_uid]]
            scores[row_idx, seen] = -np.inf

        order = np.argsort(-scores, axis=1)[:, :k]
        for row_idx, real_uid in enumerate(batch_real):
            result[real_uid] = [trainset.to_raw_iid(int(i)) for i in order[row_idx]]

    return result


def topk_bpr(model, train_matrix, user_map, item_map, user_ids, k, batch_size):
    user_to_idx = dict(zip(user_map["user_id"].astype(str), user_map["user_idx"]))
    idx_to_item = dict(zip(item_map["item_idx"], item_map["content_id"]))

    valid_users = [u for u in user_ids if u in user_to_idx]
    user_indices = np.array([user_to_idx[u] for u in valid_users])
    result = {}

    for start in range(0, len(user_indices), batch_size):
        batch_idx = user_indices[start:start + batch_size]
        batch_real = valid_users[start:start + batch_size]
        ids, _scores = model.recommend(
            batch_idx, train_matrix[batch_idx], N=k, filter_already_liked_items=True
        )
        for row_idx, real_uid in enumerate(batch_real):
            result[real_uid] = [int(idx_to_item[int(i)]) for i in ids[row_idx]]

    return result


def topk_random(trainset, user_ids, k, seed=42):
    rng = np.random.default_rng(seed)
    n_items = trainset.n_items
    result = {}
    for uid in user_ids:
        if uid not in trainset._raw2inner_id_users:
            continue
        inner_uid = trainset.to_inner_uid(uid)
        seen = set(iid for iid, _ in trainset.ur[inner_uid])
        candidates = np.array([i for i in range(n_items) if i not in seen])
        picks = rng.choice(candidates, size=min(k, len(candidates)), replace=False)
        result[uid] = [trainset.to_raw_iid(int(i)) for i in picks]
    return result


def topk_popular(trainset, user_ids, k):
    n_items = trainset.n_items
    item_counts = np.zeros(n_items, dtype=int)
    for u in range(trainset.n_users):
        for iid, _ in trainset.ur[u]:
            item_counts[iid] += 1
    order = np.argsort(-item_counts)

    result = {}
    for uid in user_ids:
        if uid not in trainset._raw2inner_id_users:
            continue
        inner_uid = trainset.to_inner_uid(uid)
        seen = set(iid for iid, _ in trainset.ur[inner_uid])
        topk = [i for i in order if i not in seen][:k]
        result[uid] = [trainset.to_raw_iid(int(i)) for i in topk]
    return result


def main():
    print("Trainset ve test kullanicilari yukleniyor...")
    svd_nmf_config = load_config("model/svd_config.yaml")
    trainset = load_trainset(svd_nmf_config["train_path"], svd_nmf_config["rating_scale"])
    relevant = load_test_relevant("model/test.csv")
    user_ids = list(relevant.keys())
    print(f"Test kullanici sayisi: {len(user_ids)}")

    all_topk = {}

    print("SVD icin siralamalar hesaplaniyor...")
    with open("model/svd_model.pkl", "rb") as f:
        svd = pickle.load(f)
    all_topk["SVD"] = topk_svd_nmf(svd, trainset, user_ids, K, BATCH_SIZE)

    print("NMF icin siralamalar hesaplaniyor...")
    with open("model/nmf_model.pkl", "rb") as f:
        nmf = pickle.load(f)
    all_topk["NMF"] = topk_svd_nmf(nmf, trainset, user_ids, K, BATCH_SIZE)

    print("BPR icin siralamalar hesaplaniyor...")
    with open("hafta6_model/data/bpr_model.pkl", "rb") as f:
        bpr = pickle.load(f)
    train_matrix = sp.load_npz("hafta6_model/data/train_matrix.npz").tocsr()
    user_map = pd.read_csv("hafta6_model/data/user_index_map.csv")
    item_map = pd.read_csv("hafta6_model/data/item_index_map.csv")
    all_topk["BPR"] = topk_bpr(bpr, train_matrix, user_map, item_map, user_ids, K, BATCH_SIZE)

    print("Rastgele baseline hesaplaniyor...")
    all_topk["Rastgele"] = topk_random(trainset, user_ids, K)

    print("En Populer baseline hesaplaniyor...")
    all_topk["En Populer"] = topk_popular(trainset, user_ids, K)

    with open("hafta6_model/data/topk_lists.pkl", "wb") as f:
        pickle.dump({"relevant": relevant, "topk": all_topk}, f)

    print()
    for name, topk in all_topk.items():
        print(f"{name}: {len(topk)} kullanici icin liste uretildi")
    print("Kaydedildi: hafta6_model/data/topk_lists.pkl")


if __name__ == "__main__":
    main()
