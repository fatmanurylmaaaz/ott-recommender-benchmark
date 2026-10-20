import sys

import polars as pl
import yaml


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "preprocessing/cold_start_config.yaml"
    config = load_config(config_path)

    df = pl.read_csv(config["input_path"])

    before_rows = len(df)
    before_users = df["user_id"].n_unique()
    before_items = df["content_id"].n_unique()
    print(f"Baslangic: {before_rows} satir, {before_users} kullanici, {before_items} icerik")

    min_user = config["min_user_interactions"]
    min_item = config["min_item_interactions"]

    # Iteratif budama: kullanici filtresi icerik sayilarini, icerik filtresi de
    # kullanici sayilarini degistirebilir (bkz. tek gecis denemesinde 1 kullanicinin
    # 3'ten 2'ye dusmesi). Satir sayisi degismeyene kadar (yakinsayana kadar)
    # sirayla kullanici -> icerik filtresini tekrar tekrar uyguluyoruz.
    round_no = 0
    while True:
        round_no += 1
        rows_before_round = len(df)

        user_counts = df.group_by("user_id").agg(pl.len().alias("n"))
        keep_users = user_counts.filter(pl.col("n") >= min_user)["user_id"]
        df = df.filter(pl.col("user_id").is_in(keep_users))

        item_counts = df.group_by("content_id").agg(pl.len().alias("n"))
        keep_items = item_counts.filter(pl.col("n") >= min_item)["content_id"]
        df = df.filter(pl.col("content_id").is_in(keep_items))

        rows_after_round = len(df)
        print(f"  tur {round_no}: {rows_before_round} -> {rows_after_round} satir")

        if rows_after_round == rows_before_round:
            break

    after_rows = len(df)
    after_users = df["user_id"].n_unique()
    after_items = df["content_id"].n_unique()

    df.write_csv(config["output_path"])

    print(f"Yakinsama: {round_no} turda tamamlandi")
    print(f"Sonuc: {after_rows} satir, {after_users} kullanici, {after_items} icerik")


if __name__ == "__main__":
    main()
