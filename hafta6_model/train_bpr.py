"""BPR (Bayesian Personalized Ranking) modelini train_matrix.npz uzerinde egitir.

model/train_svd.py ve model/train_nmf.py ile ayni desen: config oku, egit,
modeli pickle olarak kaydet, kisa bir rapor yaz.
"""
import pickle
import sys

import scipy.sparse as sp
import yaml
from implicit.bpr import BayesianPersonalizedRanking


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "hafta6_model/bpr_config.yaml"
    config = load_config(config_path)

    print("Train matrisi yukleniyor...")
    matrix = sp.load_npz(config["matrix_path"])
    print(f"Matris: {matrix.shape[0]} kullanici, {matrix.shape[1]} icerik, {matrix.nnz} pozitif etkilesim")

    print("BPR egitiliyor...")
    model = BayesianPersonalizedRanking(
        factors=config["factors"],
        learning_rate=config["learning_rate"],
        regularization=config["regularization"],
        iterations=config["iterations"],
        random_state=config.get("random_state", 42),
    )
    # BPR, matrisin degerini (hep 1 yazmistik) kullanmiyor - sadece dolu
    # hucreleri "kullanici bunu begendi" seklinde ikili (binary) bir sinyal
    # olarak okuyor (bkz. fit() docstring'i).
    model.fit(matrix, show_progress=True)

    with open(config["model_path"], "wb") as f:
        pickle.dump(model, f)

    lines = [
        "BPR MODEL EGITIMI RAPORU",
        "=" * 60,
        "",
        f"Kullanici sayisi: {matrix.shape[0]}",
        f"Icerik sayisi: {matrix.shape[1]}",
        f"Pozitif etkilesim sayisi: {matrix.nnz}",
        "",
        f"factors: {config['factors']}",
        f"learning_rate: {config['learning_rate']}",
        f"regularization: {config['regularization']}",
        f"iterations: {config['iterations']}",
        f"random_state: {config.get('random_state', 42)}",
        "",
        f"user_factors shape: {model.user_factors.shape}",
        f"item_factors shape: {model.item_factors.shape}",
    ]
    with open(config["report_path"], "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Model kaydedildi: {config['model_path']}")
    print(f"Rapor kaydedildi: {config['report_path']}")


if __name__ == "__main__":
    main()
