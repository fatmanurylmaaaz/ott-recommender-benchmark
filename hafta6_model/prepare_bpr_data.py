"""train.csv'yi BPR (implicit) icin sparse user-item matrisine cevirir.

BPR rating buyuklugunu degil, sadece etkilesim olup olmadigini kullanir -
bu yuzden rating sutunu atiliyor, her satir "pozitif etkilesim" (deger=1)
olarak matrise yaziliyor.
"""
import sys

import numpy as np
import polars as pl
import scipy.sparse as sp
import yaml


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "hafta6_model/bpr_data_config.yaml"
    config = load_config(config_path)

    df = pl.read_csv(config["train_path"])

    # user_id / content_id icin 0'dan baslayan yogun (dense) tam sayi indeksler.
    # rank("dense"), ayni degere hep ayni indeksi verir - bu sayede matris
    # satir/sutunlariyla gercek id'ler arasinda tutarli bir esleme kuruluyor.
    df = df.with_columns(
        (pl.col("user_id").rank("dense") - 1).cast(pl.Int64).alias("user_idx"),
        (pl.col("content_id").rank("dense") - 1).cast(pl.Int64).alias("item_idx"),
    )

    # Eslesme tablolari - degerlendirme asamasinda (indeks -> gercek id) geri
    # donmek icin gerekecek.
    user_map = df.select(["user_id", "user_idx"]).unique().sort("user_idx")
    item_map = df.select(["content_id", "item_idx"]).unique().sort("item_idx")
    user_map.write_csv(config["user_map_path"])
    item_map.write_csv(config["item_map_path"])

    n_users = user_map.height
    n_items = item_map.height

    # Sparse (CSR) matris: 335.926 x 1.535 boyutunda dense bir matris ~516M
    # hucre olurdu, sparse format sadece dolu (~2M) hucreyi tutuyor.
    rows = df["user_idx"].to_numpy()
    cols = df["item_idx"].to_numpy()
    data = np.ones(len(df), dtype=np.float32)
    matrix = sp.csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
    sp.save_npz(config["matrix_path"], matrix)

    print(f"Kullanici sayisi: {n_users}")
    print(f"Icerik sayisi: {n_items}")
    print(f"Dolu hucre (pozitif etkilesim) sayisi: {matrix.nnz}")
    print(f"Yogunluk (density): {matrix.nnz / (n_users * n_items) * 100:.4f}%")
    print(f"Matris kaydedildi: {config['matrix_path']}")
    print(f"Eslesme tablolari kaydedildi: {config['user_map_path']}, {config['item_map_path']}")


if __name__ == "__main__":
    main()
