import sys

import polars as pl
import yaml


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "trending/trending_config.yaml"
    config = load_config(config_path)

    df = pl.read_csv(config["input_path"])

    # Content basina ozet: Reddit'in "oy sayisi"na karsilik gelen vote_count
    # (o content'i izleyen benzersiz kullanici sayisi - "1 kullanici 1 oy"
    # mantigi; girdi tekillestirilmis oldugu icin satir sayisi = kullanici
    # sayisi) ve "guncellik" sinyali olarak en son etkilesim zamani.
    summary = df.group_by("content_id").agg(
        pl.len().alias("vote_count"),
        pl.col("last_watch").max().alias("last_activity"),
    )

    # Referans zaman: veri setindeki en son timestamp ("simdi" kabul edilir).
    reference_time = df["last_watch"].max()
    decay_seconds = config["decay_seconds"]

    # Uyarlanmis Reddit hot formulu:
    #   score = log10(max(vote_count,1)) + (last_activity - referans_zaman) / decay_seconds
    # Ikinci terim gecmis icerikler icin negatif oldugundan skoru asagi ceker;
    # decay_seconds ne kadar buyukse guncellik o kadar az agirlik tasir.
    # Esit skorlarda content_id'ye gore siralanir (cikti her calistirmada ayni olsun diye).
    summary = summary.with_columns(
        (
            pl.col("vote_count").clip(lower_bound=1).log10()
            + (pl.col("last_activity") - reference_time) / decay_seconds
        ).alias("score")
    ).sort(["score", "content_id"], descending=[True, False])

    summary.write_csv(config["output_path"])

    top_n = config.get("top_n", 20)
    print(f"Toplam content: {len(summary)}")
    print(f"Referans zaman: {reference_time}")
    print()
    print(f"En trend {top_n} icerik:")
    print(f"{'content_id':>12} {'vote_count':>11} {'last_activity':>14} {'score':>8}")
    for row in summary.head(top_n).iter_rows(named=True):
        print(
            f"{row['content_id']:>12} {row['vote_count']:>11} "
            f"{row['last_activity']:>14} {row['score']:>8.3f}"
        )


if __name__ == "__main__":
    main()
