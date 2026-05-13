import os
import sys
import time


sys.path.append(os.path.join(os.path.dirname(__file__), "src"))


def check_file_exists(path):
    return os.path.exists(path)


def run_command(command, description, skip_check=None):
    print(f"\n=== {description} ===")

    if skip_check and skip_check():
        print(f"[SKIP] {description}")
        return True

    print(f"command: {command}")
    start_time = time.time()
    result = os.system(command)
    elapsed = time.time() - start_time

    if result != 0:
        print(f"[FAIL] {description}, return code: {result}, elapsed: {elapsed:.2f}s")
        return False

    print(f"[DONE] {description}, elapsed: {elapsed:.2f}s")
    return True


def main():
    print("=== Few-shot traffic classification pipeline ===")
    print("Main path: MAE pretraining + AdapterMetricNet fine-tuning")

    # run_command(
    #     "python src/preprocess.py",
    #     "data preprocessing",
    #     skip_check=lambda: check_file_exists("data/processed/unsw_X.npy")
    #     and check_file_exists("data/processed/cicids_0_seed42.npz")
    #     and check_file_exists("data/processed/test_set.npz"),
    # )
    #
    # run_command(
    #     "python src/pretrain.py --model mae",
    #     "MAE pretraining",
    #     skip_check=lambda: check_file_exists("checkpoints/mae_pretrain.pth"),
    # )

    run_command(
        "python src/finetune.py --model all",
        "AdapterMetricNet fine-tuning with CEw + CSA-PM",
    )

    # run_command(
    #     "python src/k_shot_evaluator.py --models none mae",
    #     "k-shot evaluation",
    # )
    #
    # run_command(
    #     "python src/ablation_experiments.py",
    #     "loss ablation experiments",
    # )

    run_command(
        "python src/comparison_experiments.py",
        "traditional baseline comparison",
    )

    run_command(
        "python src/comprehensive_analysis.py",
        "comparison experiments paper figure",
    )

    #
    # run_command(
    #     "python src/cross_domain_transfer.py",
    #     "MAE-only cross-domain transfer",
    # )

    # run_command(
    #     "python src/paper_figures.py",
    #     "paper figure and table generation",
    # )

    print("\n=== Pipeline complete ===")


if __name__ == "__main__":
    main()
