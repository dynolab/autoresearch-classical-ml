import json
from pathlib import Path

import matplotlib.pyplot as plt


def plot_results(run_id: str) -> None:
    results_json = Path(run_id) / "results.json"
    iterations = []
    train_mae = []
    test_mae = []
    size = []
    comment = []
    with open(results_json, "r") as f:
        results = json.load(f)
        for iteration in results["iterations"]:
            iterations.append(iteration["i"])
            train_mae.append(iteration["train_mae"])
            test_mae.append(iteration["test_mae"])
            size.append(iteration["size"])
            comment.append(iteration["comment"] if "comment" in iteration else None)

    fig, axes = plt.subplots(3, 1, figsize=(12, 6))
    axes[0].plot(iterations, train_mae, "o--", linewidth=2, label="Train MSE")
    axes[1].plot(iterations, test_mae, "o--", linewidth=2, label="Test MSE")
    axes[2].plot(iterations, size, "o--", linewidth=2, label="Size (bytes)")
    for i in range(len(comment)):
        if comment[i] is not None:
            axes[1].annotate(comment[i], (iterations[i], test_mae[i]))

    axes[0].set_ylabel("Train MSE", fontsize=12)
    axes[1].set_ylabel("Test MSE", fontsize=12)
    axes[2].set_ylabel("Size (bytes)", fontsize=12)
    axes[2].set_xlabel("Iteration", fontsize=12)
    fig.tight_layout()
    fig.savefig(Path("images") / f"{run_id}_results.png", dpi=100)


if __name__ == "__main__":
    plot_results("run_1")
    plot_results("run_3")
