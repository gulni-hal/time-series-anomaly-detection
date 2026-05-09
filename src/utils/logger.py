"""Deney Loglama — Tüm sonuçlar CSV'ye otomatik kaydedilir."""

import os, csv, json
import numpy as np
from datetime import datetime
from collections import defaultdict


class ExperimentLogger:
    def __init__(self, log_dir: str = "logs"):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir  = log_dir
        self.records  = []
        self.run_id   = datetime.now().strftime("%Y%m%d_%H%M%S")

    def log(self, model: str, dataset: str, seed: int, scenario: str,
            metrics: dict, train_time: float = 0.0, infer_time: float = 0.0,
            extra: dict = None):
        record = {
            "run_id":     self.run_id,
            "model":      model,
            "dataset":    dataset,
            "seed":       seed,
            "scenario":   scenario,
            "f1":         metrics["f1"],
            "accuracy":   metrics["accuracy"],
            "precision":  metrics["precision"],
            "recall":     metrics["recall"],
            "train_time": round(train_time, 3),
            "infer_time": round(infer_time, 6),
        }
        if extra:
            record.update(extra)
        self.records.append(record)
        print(f"  [LOG] {model:<10} | {dataset:<5} | seed={seed} | "
              f"{scenario:<10} | F1={metrics['f1']:.4f}")

    def save(self):
        if not self.records:
            return
        csv_path  = os.path.join(self.log_dir, f"results_{self.run_id}.csv")
        json_path = os.path.join(self.log_dir, f"results_{self.run_id}.json")

        all_keys = list(dict.fromkeys(k for r in self.records for k in r.keys()))
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            for record in self.records:
                row = {k: record.get(k, "") for k in all_keys}
                writer.writerow(row)

        with open(json_path, "w") as f:
            json.dump(self.records, f, indent=2)

        print(f"\n[✓] Sonuçlar kaydedildi → {csv_path}")
        return csv_path

    def print_summary(self):
        """Tablo 1 formatında özet."""
        grouped = defaultdict(list)
        for r in self.records:
            if r["scenario"] == "original":
                grouped[(r["model"], r["dataset"])].append(r["f1"])

        print("\n" + "="*55)
        print("TABLO 1 — Model Performansı (mean F1 ± std)")
        print("="*55)
        print(f"{'Model':<12} {'Dataset':<8} {'F1 (mean ± std)'}")
        print("-"*55)
        for (model, ds), scores in sorted(grouped.items()):
            print(f"{model:<12} {ds:<8} {np.mean(scores):.4f} ± {np.std(scores):.4f}")
