# ott-recommender-benchmark

OTT izleme verisi üzerinde SVD, NMF ve BPR tabanlı öneri modellerinin karşılaştırması:
veri hazırlama pipeline'ı ve 8 metrikle (Precision, Recall, HitRate, NDCG, MRR,
Coverage, Diversity, Gini) değerlendirme.

> Veri seti ve model dosyaları repoya dahil değildir; repo kodun incelenmesi içindir.

## Klasörler

| Klasör | İçerik |
|---|---|
| `preprocessing/` | Ham veriden eğitime hazır veri setine 4 adım: filtreleme, tekilleştirme, rating türetme, cold-start filtresi |
| `model/` | Zaman bazlı train/test ayrımı, SVD ve NMF eğitimi, ranking değerlendirmesi |
| `hafta6_model/` | BPR eğitimi, hiperparametre denemesi, popülerlik hibriti, 5 yöntemin 8 metrikle karşılaştırması (`reports/`, `charts/`) |
| `trending/` | Trend içerik tespiti (Reddit hot uyarlaması ve velocity), ana pipeline'dan bağımsız yan çalışma |
| `rating_analysis/` | Rating (izlenme tamamlanma oranı) dağılımı |
| `pandas/` | Tekilleştirme adımının ilk pandas sürümü (Polars sürümüyle karşılaştırma için) |

Her script kendi `*_config.yaml` dosyasını okur ve çıktısını yeni bir dosyaya yazar.
Scriptler proje kök dizininden çalıştırılır.

## Çalıştırma

```bash
pip install -r requirements.txt   # Python 3.8.10 ile test edildi

# 1. Veri hazırlama
python preprocessing/filter_data.py
python preprocessing/aggregate_data_polars.py
python preprocessing/derive_rating.py
python preprocessing/filter_cold_start.py

# 2. SVD / NMF
python model/prepare_split.py
python model/train_svd.py
python model/train_nmf.py
python model/eval_precision_recall.py

# 3. BPR ve 8 metrik değerlendirme
python hafta6_model/prepare_bpr_data.py
python hafta6_model/train_bpr.py
python hafta6_model/generate_topk_lists.py
python hafta6_model/plot_precision_recall.py
python hafta6_model/plot_ndcg_mrr.py
python hafta6_model/plot_hitrate_comparison.py
python hafta6_model/eval_diversity.py

# Deneyler: tune_bpr.py (hiperparametre), hybrid_bpr.py (BPR + popülerlik)
```

## Sonuç

- SVD/NMF rating tahmininde makul hata verse de (RMSE ~0,31) top-N öneride rastgele
  öneriden bile zayıf kaldı (HitRate@10 < 0,01).
- Sıralamayı doğrudan öğrenen BPR, HitRate@10'u 0,27'ye çıkardı.
- En Popüler baseline'ı isabette önde (HitRate@10 0,46) ama kataloğun yalnızca %2'sini
  öneriyor; BPR kataloğun %83'ünü kullanıyor (Gini 0,99'a karşı 0,76).
