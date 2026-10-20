import sys

import pandas as pd
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


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def dominant(series):
    # Bir (user_id, content_id) çiftinin tekrarlarında en sık görülen değer.
    return series.value_counts().idxmax()


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "pandas/aggregate_config_pandas.yaml"
    config = load_config(config_path)

    # keep_default_na=False: bos alanlar (ör. eksik lang) "" olarak kalsin,
    # NaN'a donusmesin (filter_data.py'deki bos deger davranisiyla tutarli).
    df = pd.read_csv(
        config["input_path"],
        header=None,
        names=COLUMNS,
        dtype=str,
        keep_default_na=False,
    )
    df["timestamp"] = df["timestamp"].astype(int)
    df["duration"] = df["duration"].astype(int)
    df["watched_duration"] = df["watched_duration"].astype(int)

    grouped = (
        df.groupby(["user_id", "content_id"])
        .agg(
            type=("type", "first"),
            duration=("duration", "max"),
            watched_duration=("watched_duration", "max"),
            first_watch=("timestamp", "min"),
            last_watch=("timestamp", "max"),
            watch_count=("timestamp", "count"),
            device_id=("device_id", dominant),
            lang=("lang", dominant),
            country=("country", dominant),
        )
        .reset_index()
    )

    grouped["active_duration_seconds"] = grouped["last_watch"] - grouped["first_watch"]

    grouped = grouped[
        [
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
    ]

    grouped.to_csv(config["output_path"], index=False)

    print(f"Girdi satir sayisi: {len(df)}")
    print(f"Cikti (benzersiz user-content cifti) satir sayisi: {len(grouped)}")


if __name__ == "__main__":
    main()
