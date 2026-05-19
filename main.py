"""
Main pipeline for IR-SCB-Focal experiments.

Default order:
  1. Optional preprocessing
  2. Optional MAE pretraining
  3. Main in-domain/cross-domain comparison
  4. Focal-family ablation
  5. Paper tables and figures
"""

import argparse
import os
import subprocess
import sys
import time


ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable


def file_exists(path):
    return os.path.exists(os.path.join(ROOT, path))


def run_step(command, description, skip=False):
    print(f"\n=== {description} ===")
    if skip:
        print(f"[SKIP] {description}")
        return True

    print("command:", " ".join(command))
    start = time.time()
    result = subprocess.run(command, cwd=ROOT)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"[FAIL] {description}, return code={result.returncode}, elapsed={elapsed:.2f}s")
        return False

    print(f"[DONE] {description}, elapsed={elapsed:.2f}s")
    return True


def build_common_args(args):
    common = [
        "--domain",
        args.domain,
        "--ratios",
        *[str(ratio) for ratio in args.ratios],
        "--seeds",
        *[str(seed) for seed in args.seeds],
        "--epochs",
        str(args.epochs),
        "--rho-min",
        str(args.rho_min),
        "--rho-max",
        str(args.rho_max),
        "--ir-ref",
        str(args.ir_ref),
        "--ratio-decay-ref",
        str(args.ratio_decay_ref),
        "--output-dir",
        args.output_dir,
    ]
    return common


def main():
    parser = argparse.ArgumentParser(description="Run the IR-SCB-Focal experiment pipeline")
    parser.add_argument("--domain", choices=["indomain", "crossdomain", "all"], default="all")
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.1, 1, 5, 10])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 52, 62])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--rho-min", type=float, default=0.25)
    parser.add_argument("--rho-max", type=float, default=0.5)
    parser.add_argument("--ir-ref", type=float, default=100000.0)
    parser.add_argument("--ratio-decay-ref", type=float, default=5.0)
    parser.add_argument("--output-dir", type=str, default="results/ir_scb_focal")
    parser.add_argument("--preprocess", action="store_true", help="run preprocessing before experiments")
    parser.add_argument("--pretrain", action="store_true", help="run MAE pretraining before experiments")
    parser.add_argument("--skip-main", action="store_true", help="skip main comparison experiments")
    parser.add_argument("--skip-ablation", action="store_true", help="skip ablation experiments")
    parser.add_argument("--skip-figures", action="store_true", help="skip paper figure/table generation")
    parser.add_argument("--force-pretrain", action="store_true", help="run pretraining even if checkpoint exists")
    args = parser.parse_args()

    print("=== IR-SCB-Focal few-shot traffic classification pipeline ===")
    print(
        f"domain={args.domain}, ratios={args.ratios}, seeds={args.seeds}, "
        f"epochs={args.epochs}, rho_min={args.rho_min}, "
        f"rho_max={args.rho_max}, ir_ref={args.ir_ref}, "
        f"ratio_decay_ref={args.ratio_decay_ref}, output_dir={args.output_dir}"
    )

    if args.preprocess:
        ok = run_step(
            [PYTHON, "src/preprocess.py"],
            "data preprocessing",
            skip=(
                file_exists("data/processed/unsw_X.npy")
                and file_exists("data/processed/test_set.npz")
                and file_exists("data/processed/cicids_0_seed42.npz")
            ),
        )
        if not ok:
            sys.exit(1)

    if args.pretrain:
        ok = run_step(
            [PYTHON, "src/pretrain.py"],
            "MAE pretraining",
            skip=file_exists("checkpoints/mae_pretrain.pth") and not args.force_pretrain,
        )
        if not ok:
            sys.exit(1)

    common_args = build_common_args(args)

    if not args.skip_main:
        ok = run_step(
            [PYTHON, "src/fewshot_algorithm_experiments.py", *common_args],
            "main algorithm comparison",
        )
        if not ok:
            sys.exit(1)

    if not args.skip_ablation:
        ok = run_step(
            [PYTHON, "src/ablation_experiments.py", *common_args],
            "Focal-family ablation experiments",
        )
        if not ok:
            sys.exit(1)

    if not args.skip_figures:
        ok = run_step(
            [PYTHON, "src/paper_figures.py"],
            "paper table and figure generation",
        )
        if not ok:
            sys.exit(1)

    print("\n=== Pipeline complete ===")


if __name__ == "__main__":
    main()
