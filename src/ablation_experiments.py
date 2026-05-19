"""
Ablation experiments for IR-SCB-Focal.

This file is the only entry for ablation experiments.

Compared variants:
  - Focal
  - CB-Focal
  - SCB-Focal rho=0.25
  - SCB-Focal rho=0.5
  - IR-SCB-Focal
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np

from finetune_methods import (
    ABLATION_METHODS,
    EPOCHS,
    IR_REF,
    METHOD_LABELS,
    RATIO_DECAY_REF,
    RHO_MIN,
    RHO_MAX,
    load_fewshot_set,
    load_test_set,
    ratio_token,
    train_one_run,
)


OUTPUT_DIR = "results/ir_scb_focal"


def json_safe(obj):
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def prediction_path(output_dir, domain_mode, method, ratio, seed):
    return os.path.join(
        output_dir,
        "predictions",
        f"{domain_mode}_ablation_{method}_ratio{ratio_token(ratio)}_seed{seed}.npz",
    )


def strip_predictions(run, pred_path):
    run = dict(run)
    predictions = run.pop("predictions")
    np.savez_compressed(pred_path, y_true=predictions["y_true"], y_pred=predictions["y_pred"])
    run["prediction_path"] = pred_path
    return run


def summarize_runs(runs):
    grouped = defaultdict(list)
    for run in runs:
        grouped[(str(run["ratio"]), run["method"])].append(run)

    summary = {}
    for (ratio, method), items in grouped.items():
        metric_summary = {"runs": int(len(items)), "method_label": METHOD_LABELS.get(method, method)}
        for metric in ["macro_f1", "accuracy", "balanced_accuracy", "rare_recall", "weighted_f1"]:
            values = np.asarray([item["test"][metric] for item in items], dtype=np.float64)
            metric_summary[f"{metric}_mean"] = float(values.mean())
            metric_summary[f"{metric}_std"] = float(values.std(ddof=0))
            metric_summary[f"{metric}_list"] = values.tolist()
        summary.setdefault(ratio, {})[method] = metric_summary
    return summary


def run_ablation_group(domain_mode, ratios, seeds, epochs, rho_min, rho_max, ir_ref, ratio_decay_ref, output_dir):
    test_X, test_y, test_path = load_test_set()
    results = {
        "config": {
            "domain_mode": domain_mode,
            "experiment": "ablation",
            "methods": list(ABLATION_METHODS),
            "method_labels": {method: METHOD_LABELS.get(method, method) for method in ABLATION_METHODS},
            "ratios": [float(ratio) for ratio in ratios],
            "seeds": [int(seed) for seed in seeds],
            "epochs": int(epochs),
            "rho_min": float(rho_min),
            "rho_max": float(rho_max),
            "ir_ref": float(ir_ref),
            "ratio_decay_ref": float(ratio_decay_ref),
            "output_dir": output_dir,
            "test_set": test_path,
        },
        "results": {},
        "summary": {},
    }
    flat_runs = []
    os.makedirs(os.path.join(output_dir, "predictions"), exist_ok=True)

    for ratio in ratios:
        ratio_key = str(ratio)
        results["results"][ratio_key] = {}
        for seed in seeds:
            train_X, train_y, train_path = load_fewshot_set(ratio, seed)
            seed_runs = []
            print(f"\n[{domain_mode}/ablation] ratio={ratio}% seed={seed} train={train_path}")

            for method in ABLATION_METHODS:
                run = train_one_run(
                    train_X=train_X,
                    train_y=train_y,
                    test_X=test_X,
                    test_y=test_y,
                    method_name=method,
                    domain_mode=domain_mode,
                    ratio=ratio,
                    seed=seed,
                    epochs=epochs,
                    rho_min=rho_min,
                    rho_max=rho_max,
                    ir_ref=ir_ref,
                    ratio_decay_ref=ratio_decay_ref,
                )
                pred_path = prediction_path(output_dir, domain_mode, method, ratio, seed)
                run = strip_predictions(run, pred_path)
                seed_runs.append(run)
                flat_runs.append(run)
                print(
                    f"  {METHOD_LABELS.get(method, method):<22} "
                    f"macro_f1={run['test']['macro_f1']:.4f} "
                    f"bacc={run['test']['balanced_accuracy']:.4f} "
                    f"rho={run['loss_info'].get('rho')}"
                )

            results["results"][ratio_key][str(seed)] = {
                "train_set": train_path,
                "runs": seed_runs,
            }

    results["summary"] = summarize_runs(flat_runs)
    return results


def save_results(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, default=json_safe, indent=2)
    print(f"saved ablation results: {path}")


def main():
    parser = argparse.ArgumentParser(description="Run IR-SCB-Focal ablation experiments")
    parser.add_argument("--domain", choices=["indomain", "crossdomain", "all"], default="all")
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.1, 1, 5, 10])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 52, 62])
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--rho-min", type=float, default=RHO_MIN)
    parser.add_argument("--rho-max", type=float, default=RHO_MAX)
    parser.add_argument("--ir-ref", type=float, default=IR_REF)
    parser.add_argument("--ratio-decay-ref", type=float, default=RATIO_DECAY_REF)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    domain_modes = ["indomain", "crossdomain"] if args.domain == "all" else [args.domain]
    for domain_mode in domain_modes:
        results = run_ablation_group(
            domain_mode=domain_mode,
            ratios=tuple(args.ratios),
            seeds=tuple(args.seeds),
            epochs=args.epochs,
            rho_min=args.rho_min,
            rho_max=args.rho_max,
            ir_ref=args.ir_ref,
            ratio_decay_ref=args.ratio_decay_ref,
            output_dir=args.output_dir,
        )
        save_results(results, os.path.join(args.output_dir, f"{domain_mode}_ablation_results.json"))


if __name__ == "__main__":
    main()
