import os, csv, json
import numpy as np
from datetime import datetime
from collections import defaultdict


class ExperimentLogger:
    def __init__(self, log_dir="logs"):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.records = []
        self.run_id  = datetime.now().strftime("%Y%m%d_%H%M%S")

    def log(self, model, dataset, seed, scenario, metrics,
            train_time=0.0, infer_time=0.0, extra=None):
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
        print("  [LOG] " + model + " | " + dataset +
              " | seed=" + str(seed) + " | " + scenario +
              " | F1=" + str(metrics["f1"]))

    def save(self):
        if not self.records:
            return
        csv_path  = os.path.join(self.log_dir, "results_" + self.run_id + ".csv")
        json_path = os.path.join(self.log_dir, "results_" + self.run_id + ".json")
        all_keys  = list(dict.fromkeys(k for r in self.records for k in r.keys()))

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            for record in self.records:
                writer.writerow({k: record.get(k, "") for k in all_keys})

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=2)

        print("\n[OK] Results saved: " + csv_path)
        return csv_path

    def print_summary(self):
        grouped = defaultdict(list)
        for r in self.records:
            if r["scenario"] == "original":
                grouped[(r["model"], r["dataset"])].append(r["f1"])

        print("\n" + "="*55)
        print("TABLE 1 - Model Performance (mean F1 +/- std)")
        print("="*55)
        print("Model        Dataset  F1")
        print("-"*55)
        for (model, ds), scores in sorted(grouped.items()):
            print(model + " " * (12-len(model)) + ds + " " * (8-len(ds)) +
                  str(round(np.mean(scores), 4)) + " +/- " + str(round(np.std(scores), 4)))