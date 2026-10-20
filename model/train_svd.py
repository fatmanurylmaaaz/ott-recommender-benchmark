import pickle
import sys

from surprise import SVD, accuracy

from common import load_config, load_testset, load_trainset


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "model/svd_config.yaml"
    config = load_config(config_path)

    print("Trainset/testset yukleniyor...")
    trainset = load_trainset(config["train_path"], config["rating_scale"])
    testset = load_testset(config["test_path"])
    print(f"Train: {trainset.n_ratings} etkilesim, {trainset.n_users} kullanici, {trainset.n_items} icerik")
    print(f"Test: {len(testset)} etkilesim")

    seed = config.get("random_state", 42)
    print("SVD egitiliyor...")
    algo = SVD(random_state=seed)
    algo.fit(trainset)  # train.csv'deki gozlemlenen etkilesimlerden latent factor'leri ogrenir
    predictions = algo.test(testset)  # test.csv'deki her satir icin rating tahmini uretir
    # RMSE buyuk hatalari agir cezalandirir, MAE her hatayi esit agirlikta sayar -
    # ikisini birlikte raporlamak modelin farkli acilardan ne kadar iyi oldugunu gosterir.
    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)
    print(f"SVD -> RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    # Egitilmis modeli pickle ile diske yaz - ileride yeniden egitmeden
    # (.predict() ile) tahmin uretmek icin tekrar yuklenebilir.
    with open(config["model_path"], "wb") as f:
        pickle.dump(algo, f)

    lines = [
        "SVD MODEL EGITIMI RAPORU",
        "=" * 60,
        "",
        f"Train: {trainset.n_ratings} etkilesim, {trainset.n_users} kullanici, {trainset.n_items} icerik",
        f"Test: {len(testset)} etkilesim (zaman bazli leave-last-out split)",
        f"Rating araligi: {tuple(config['rating_scale'])}",
        f"Random seed: {seed}",
        "",
        f"RMSE: {rmse:.4f}",
        f"MAE:  {mae:.4f}",
        "",
        "Ilgili diger rapor: rapor_nmf.txt",
    ]
    # trending/ klasorundeki rapor deseniyle ayni: her yontemin kendi rapor
    # dosyasi var, diger rapora capraz referans veriyor.
    with open(config["report_path"], "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Rapor kaydedildi: {config['report_path']}")
    print(f"Model kaydedildi: {config['model_path']}")


if __name__ == "__main__":
    main()
