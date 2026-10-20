import csv
import operator
import sys

import yaml

# CSV'de başlık satırı yok, bu yüzden sütun sırasını burada sabitliyoruz.
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

# Config'teki operator string'ini (">" gibi) karşılaştırma fonksiyonuna çevirir.
OPERATORS = {
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def row_passes(row, filters, rules):
    # Kategorik sütun filtreleri: include verilmişse sadece o değerler,
    # exclude verilmişse o değerler haric her şey tutulur.
    for col, rule in filters.items():
        val = row[COLUMNS.index(col)]
        if "include" in rule and val not in rule["include"]:
            return False
        if "exclude" in rule and val in rule["exclude"]:
            return False

    # İki sütunu karşılaştıran kurallar (ör. watched_duration > duration
    # gibi veri hatalarını elemek için). Kural sağlanırsa satır atılır.
    for rule in rules:
        val_a = int(row[COLUMNS.index(rule["column_a"])])
        val_b = int(row[COLUMNS.index(rule["column_b"])])
        if OPERATORS[rule["operator"]](val_a, val_b):
            return False

    return True


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "preprocessing/filter_config.yaml"
    config = load_config(config_path)
    filters = config.get("filters", {})
    rules = config.get("rules", [])

    total = 0
    kept = 0
    # Satır satır okuyup yazıyoruz; dosya 185 MB olduğu için tamamını
    # belleğe almak yerine stream ediyoruz. Orijinal dosya sadece okunur,
    # sonuç ayrı bir dosyaya (output_path) yazılır.
    with open(config["input_path"], encoding="utf-8", newline="") as fin, open(
        config["output_path"], "w", encoding="utf-8", newline=""
    ) as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        for row in reader:
            total += 1
            if row_passes(row, filters, rules):
                writer.writerow(row)
                kept += 1

    print(f"Toplam satir: {total}")
    print(f"Tutulan satir: {kept}")
    print(f"Atilan satir: {total - kept}")


if __name__ == "__main__":
    main()
