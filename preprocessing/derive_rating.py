import sys

import polars as pl
import yaml


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "preprocessing/derive_rating_config.yaml"
    config = load_config(config_path)

    df = pl.read_csv(config["input_path"])

    # rating: icerigin ne kadarinin izlendigi (0-1 arasi tamamlama orani).
    # Veride duration=0 olan satir olmadigi dogrulandi, bu yuzden ekstra guvenlik
    # kontrolu eklenmedi (varsa ileride burada ele alinmali).
    df = df.with_columns(
        (pl.col("watched_duration") / pl.col("duration")).alias("rating")
    )

    before = len(df)
    min_rating = config.get("min_rating")
    if min_rating is not None:
        df = df.filter(pl.col("rating") >= min_rating)

    df.write_csv(config["output_path"])

    stats = df.select(
        pl.col("rating").min().alias("min"),
        pl.col("rating").max().alias("max"),
        pl.col("rating").mean().alias("mean"),
        pl.col("rating").median().alias("median"),
    ).row(0, named=True)

    print(f"Girdi satir sayisi: {before}")
    if min_rating is not None:
        print(f"Gurultu filtresi (rating < {min_rating}) sonrasi atilan satir: {before - len(df)}")
    print(f"Cikti satir sayisi: {len(df)}")
    print(f"Rating min: {stats['min']:.4f}")
    print(f"Rating max: {stats['max']:.4f}")
    print(f"Rating ortalama: {stats['mean']:.4f}")
    print(f"Rating medyan: {stats['median']:.4f}")


if __name__ == "__main__":
    main()
