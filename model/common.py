"""SVD ve NMF egitim scriptleri arasinda paylasilan veri yukleme yardimcilari."""
import pandas as pd
import yaml
from surprise import Dataset, Reader


def load_config(path):
    # Diger scriptlerdeki ile ayni: YAML config dosyasini dict'e cevirir.
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_trainset(train_path, rating_scale):
    # train.csv'yi Surprise'in beklendigi Trainset formatina cevirir.
    df = pd.read_csv(train_path)
    # Reader, rating'in hangi aralikta oldugunu (0-1) Surprise'a bildirir.
    reader = Reader(rating_scale=tuple(rating_scale))
    # Dataset.load_from_df, sutun sirasi onemli olacak sekilde
    # (user, item, rating) uclusunu Surprise formatina sokar.
    data = Dataset.load_from_df(df[["user_id", "content_id", "rating"]], reader)
    # build_full_trainset(): verinin tamami egitim icin kullanilacak
    # (test.csv zaten ayri tutuldugu icin burada ekstra bolme yapilmiyor).
    return data.build_full_trainset()


def load_testset(test_path):
    df = pd.read_csv(test_path)
    # Surprise'in test() metodu (raw_uid, raw_iid, gercek_rating) uclulerinden
    # olusan bir liste bekliyor.
    return list(df[["user_id", "content_id", "rating"]].itertuples(index=False, name=None))
