import sys

import polars as pl
import yaml


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "model/split_config.yaml"
    config = load_config(config_path)

    df = pl.read_csv(config["input_path"])

    # Her kullanicinin en son etkilesimini (last_watch'a gore) bul; sadece
    # o satirlar teste, geri kalani train'e gidecek. Ayni kullanicinin
    # birden fazla "en son" satiri varsa (esitlik) hepsi teste dusebilir -
    # bu, sonradan kontrol edilecek kucuk bir edge-case.
    last_per_user = df.group_by("user_id").agg(pl.col("last_watch").max().alias("max_last_watch"))

    df = df.join(last_per_user, on="user_id")
    test = df.filter(pl.col("last_watch") == pl.col("max_last_watch"))
    train = df.filter(pl.col("last_watch") != pl.col("max_last_watch"))

    # Surprise sadece user_id, content_id, rating uclusunu kullanacak.
    cols = ["user_id", "content_id", "rating"]
    train.select(cols).write_csv(config["train_path"])
    test.select(cols).write_csv(config["test_path"])

    print(f"Toplam satir: {len(df)}")
    print(f"Train satir: {len(train)}")
    print(f"Test satir: {len(test)}")
    print(f"Test/toplam kullanici sayisi: {test['user_id'].n_unique()} / {df['user_id'].n_unique()}")

    # Esitlik (ayni kullanicinin birden fazla "en son" satiri) yuzunden
    # test setinde kullanici basina 1'den fazla satir olup olmadigini kontrol et.
    dup = test.group_by("user_id").agg(pl.len().alias("n")).filter(pl.col("n") > 1)
    print(f"Test setinde birden fazla satiri olan kullanici sayisi (esitlik durumu): {len(dup)}")


if __name__ == "__main__":
    main()
