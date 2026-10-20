"""train_matrix.npz'yi acip temel bilgilerini gosteren kucuk bir inceleme scripti.

Nereden calistirilirsa calistirilsin (proje koku, hafta6_model/ icinden, IDE'nin
Run butonu vb.) calisir - dosya yolu, script'in kendi konumuna gore kuruluyor.
"""
from pathlib import Path

import scipy.sparse as sp

matrix_path = Path(__file__).parent / "data" / "train_matrix.npz"
matrix = sp.load_npz(matrix_path)

print("Matris tipi/format:", repr(matrix))
print()
print("shape:", matrix.shape, " dtype:", matrix.dtype, " nnz:", matrix.nnz)
print()

print("--- Ilk 5 kullanicinin (satirin) hangi icerikleri (sutun indeksi) izledigi ---")
for u in range(5):
    row = matrix.getrow(u)
    cols = row.indices
    print(f"user_idx={u}: item_idx={list(cols)}  (toplam {len(cols)} icerik)")
print()

print("--- Kucuk bir alt-blok (ilk 8 kullanici x ilk 15 icerik), dense gorunum ---")
sub = matrix[:8, :15].toarray().astype(int)
print(sub)
