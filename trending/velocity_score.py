import sys

import polars as pl
import yaml


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "trending/velocity_config.yaml"
    config = load_config(config_path)

    df = pl.read_csv(config["input_path"])

    # Referans zaman: veri setindeki en son timestamp ("simdi" kabul edilir),
    # Reddit-uyarlamasindaki (trending_score.py) mantikla ayni.
    reference_time = df["last_watch"].max()
    recent_seconds = config["recent_window_days"] * 86400
    baseline_seconds = config["baseline_window_days"] * 86400

    recent_start = reference_time - recent_seconds
    baseline_start = recent_start - baseline_seconds

    # Her satirin last_watch'ini "bu etkilesim ne zaman oldu" olarak kullanip
    # iki pencereye ayiriyoruz: son N gun (recent) ve ondan onceki N gun (baseline).
    recent_counts = (
        df.filter(pl.col("last_watch") > recent_start)
        .group_by("content_id")
        .agg(pl.len().alias("recent_count"))
    )
    baseline_counts = (
        df.filter(
            (pl.col("last_watch") > baseline_start) & (pl.col("last_watch") <= recent_start)
        )
        .group_by("content_id")
        .agg(pl.len().alias("baseline_count"))
    )

    # Bir content sadece tek pencerede gorunebilecegi icin full (outer) join,
    # eksik taraf 0 ile dolduruluyor.
    summary = recent_counts.join(baseline_counts, on="content_id", how="full", coalesce=True)
    summary = summary.with_columns(
        pl.col("recent_count").fill_null(0),
        pl.col("baseline_count").fill_null(0),
    )

    # Gurultu filtresi: cok az recent etkilesimi olan content'ler elenir
    # (ör. 1 kisi/0 kisi gibi anlamsiz oranlar yuksek velocity vermesin diye).
    min_recent = config["min_recent_count"]
    before = len(summary)
    summary = summary.filter(pl.col("recent_count") >= min_recent)

    # velocity = son donem / onceki donem (+1, sifira bolmeyi onlemek icin).
    # Esit velocity'lerde content_id'ye gore siralanir; boylece cikti her
    # calistirmada ayni olur (group_by/join sirasi garanti degil).
    summary = summary.with_columns(
        (pl.col("recent_count") / (pl.col("baseline_count") + 1)).alias("velocity")
    ).sort(["velocity", "content_id"], descending=[True, False])

    summary.write_csv(config["output_path"])

    print(f"Referans zaman: {reference_time}")
    print(f"Son pencere: {recent_start} - {reference_time} ({config['recent_window_days']} gun)")
    print(
        f"Onceki pencere: {baseline_start} - {recent_start} "
        f"({config['baseline_window_days']} gun)"
    )
    print(f"Gurultu filtresi (recent_count < {min_recent}) sonrasi: {before} -> {len(summary)} content")
    print()
    top_n = config.get("top_n", 20)
    print(f"En yuksek velocity'li {top_n} icerik:")
    print(f"{'content_id':>12} {'recent':>7} {'baseline':>9} {'velocity':>9}")
    for row in summary.head(top_n).iter_rows(named=True):
        print(
            f"{row['content_id']:>12} {row['recent_count']:>7} "
            f"{row['baseline_count']:>9} {row['velocity']:>9.2f}"
        )


if __name__ == "__main__":
    main()
