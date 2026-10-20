"""8 metrigin (Precision, Recall, HitRate, MRR, NDCG, Coverage, Diversity,
Gini-Index) hesaplanmasinda ortak kullanilacak yardimci fonksiyonlar."""
import json

import pandas as pd


def load_test_relevant(test_path):
    """user_id (str) -> test.csv'deki gercek content_id (int) kumesi."""
    df = pd.read_csv(test_path)
    relevant = {}
    for row in df.itertuples(index=False):
        relevant.setdefault(str(row.user_id), set()).add(int(row.content_id))
    return relevant


def load_genre_map(json_path):
    """content_id (int) -> genre kumesi (set[str]). Diversity icin gerekli.

    all (1).json'daki anahtarlar string content_id, degerler icinde
    'Genres' listesi olan bir dict - genre bilgisi olmayan icerikler
    bos kume ile temsil edilir.
    """
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
