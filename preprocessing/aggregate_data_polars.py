import sys

import polars as pl
import yaml

# filter_data.py'deki ile aynı sütun sırası (CSV'de başlık yok).
COLUMNS = [
    "user_id",
    "content_id",
    "type",
    "timestamp",
    "duration",
    "watched_duration",
    "app_id",
    "device_id",
    "lang",
    "country",
]

OUTPUT_COLUMNS = [
    "user_id",
    "content_id",
    "type",
    "duration",
    "watched_duration",
    "first_watch",
    "last_watch",
    "watch_count",
    "active_duration_seconds",
    "device_id",
    "lang",
    "country",
]


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "preprocessing/aggregate_config_polars.yaml"
    config = load_config(config_path)

    # missing_utf8_is_empty_string=True: bos alanlar (ör. eksik lang) "" olarak
    # kalsin, null'a donusmesin (pandas versiyonundaki davranisla tutarli).
    df = pl.read_csv(
        config["input_path"],
        has_header=False,
        new_columns=COLUMNS,
        schema_overrides={
            "timestamp": pl.Int64,
            "duration": pl.Int64,
            "watched_duration": pl.Int64,
        },
        missing_utf8_is_empty_string=True,
    )

    # Ayni (user_id, content_id) ciftine ait tekrar eden satirlari tek satira
    # indiriyoruz. Her sutun icin secilen kural:
    #  - type: sabit kabul edilip ilk deger alinir
    #  - duration/watched_duration: max (sum yerine; ayni ciftin tekrarlarinda
    #    cift sayim yapmamak icin, watched_duration en ileri izleme noktasini gosterir)
    #  - timestamp: hem min (first_watch) hem max (last_watch) hem sayisi (watch_count)
    #  - device_id/lang/country: en sik gorulen (mode) deger; esitlikte ilk gorulen secilir
    grouped = (
        df.group_by(["user_id", "content_id"])
        .agg(
            pl.col("type").first().alias("type"),
            pl.col("duration").max().alias("duration"),
            pl.col("watched_duration").max().alias("watched_duration"),
            pl.col("timestamp").min().alias("first_watch"),
            pl.col("timestamp").max().alias("last_watch"),
            pl.col("timestamp").count().alias("watch_count"),
            pl.col("device_id").mode().first().alias("device_id"),
            pl.col("lang").mode().first().alias("lang"),
            pl.col("country").mode().first().alias("country"),
        )
        .with_columns(
            (pl.col("last_watch") - pl.col("first_watch")).alias("active_duration_seconds")
        )
        .select(OUTPUT_COLUMNS)
    )

    grouped.write_csv(config["output_path"])

    print(f"Girdi satir sayisi: {len(df)}")
    print(f"Cikti (benzersiz user-content cifti) satir sayisi: {len(grouped)}")


if __name__ == "__main__":
    main()
