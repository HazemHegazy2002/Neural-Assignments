import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from step1_data import train_loader, test_loader
from step2_lenet import LeNet5
from step3_attention import LeNet5WithAttention
from step4_train import train_model

# ─────────────────────────────────────────────
#  HYPERPARAMETERS
# ─────────────────────────────────────────────
EPOCHS       = 20
LR           = 0.001
WEIGHT_DECAY = 1e-4
PATIENCE     = 8
RESULTS_DIR  = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def save_confusion_matrix(cm, accuracy, title, save_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=range(10), yticklabels=range(10),
        linewidths=0.5, ax=ax
    )
    ax.set_title(f"{title}\nAccuracy: {accuracy:.1f}%", fontsize=13)
    ax.set_xlabel("Predicted Digit", fontsize=11)
    ax.set_ylabel("True Digit",      fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved: {save_path}")


def main():

    # ── PARAMETER COUNT ──────────────────────────────────────
    model_base_temp = LeNet5()
    model_att_temp  = LeNet5WithAttention()
    params_base = sum(p.numel() for p in model_base_temp.parameters())
    params_att  = sum(p.numel() for p in model_att_temp.parameters())
    print(f"\nLeNet-5 (No Attention)   parameters: {params_base:,}")
    print(f"LeNet-5 (With Attention) parameters: {params_att:,}")

    # ── TRAIN WITHOUT ATTENTION ──────────────────────────────
    model_base = LeNet5()
    (acc_base, tr_base, te_base,
     losses_base, per_digit_base, cm_base) = train_model(
        model_base, train_loader, test_loader,
        epochs=EPOCHS, lr=LR,
        weight_decay=WEIGHT_DECAY,
        patience=PATIENCE,
        description="LeNet-5 (No Attention)"
    )

    save_confusion_matrix(
        cm_base, acc_base,
        title="Part (a) — LeNet-5 (No Attention)",
        save_path=f"{RESULTS_DIR}/part_a_cm_no_attention.png"
    )

    # ── TRAIN WITH ATTENTION ─────────────────────────────────
    model_att = LeNet5WithAttention()
    (acc_att, tr_att, te_att,
     losses_att, per_digit_att, cm_att) = train_model(
        model_att, train_loader, test_loader,
        epochs=EPOCHS, lr=LR,
        weight_decay=WEIGHT_DECAY,
        patience=PATIENCE,
        description="LeNet-5 (With Spatial Attention)"
    )

    save_confusion_matrix(
        cm_att, acc_att,
        title="Part (a) — LeNet-5 (With Spatial Attention)",
        save_path=f"{RESULTS_DIR}/part_a_cm_with_attention.png"
    )

    # ── COMPARISON TABLE ─────────────────────────────────────
    print("\n" + "="*70)
    print("PART (a) COMPARISON TABLE — ReducedMNIST")
    print("="*70)
    print(f"{'Metric':<30} {'No Attention':>18} {'With Attention':>18}")
    print("-"*70)
    print(f"{'Accuracy (%)':<30} {acc_base:>18.1f} {acc_att:>18.1f}")
    print(f"{'Training Time (ms)':<30} {tr_base:>18.1f} {tr_att:>18.1f}")
    print(f"{'Testing Time (ms)':<30} {te_base:>18.1f} {te_att:>18.1f}")
    print(f"{'Parameters':<30} {params_base:>18,} {params_att:>18,}")
    print("-"*70)
    print(f"\n  Per-Digit Accuracy:")
    print(f"  {'Digit':<8} {'No Attention':>14} {'With Attention':>16}")
    print(f"  {'-'*40}")
    for digit in range(10):
        a = per_digit_base.get(str(digit), 0.0)
        b = per_digit_att.get(str(digit), 0.0)
        print(f"  Digit {digit}  {a:>13.1f}%  {b:>14.1f}%")
    print("="*70)

    # ── LOSS CURVES ──────────────────────────────────────────
    plt.figure(figsize=(10, 5))
    plt.plot(losses_base, label='No Attention',   color='blue')
    plt.plot(losses_att,  label='With Attention', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Part (a) — LeNet-5: Training Loss With vs Without Attention')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/part_a_loss_curves.png", dpi=150)
    plt.show()
    print(f"\n  Loss curves saved: {RESULTS_DIR}/part_a_loss_curves.png")

    # ── SAVE RESULTS TO JSON ─────────────────────────────────
    results = {
        "part_a_mnist": {
            "hyperparameters": {
                "epochs":        EPOCHS,
                "learning_rate": LR,
                "weight_decay":  WEIGHT_DECAY,
                "batch_size":    64,
                "patience":      PATIENCE,
                "optimizer":     "Adam",
                "scheduler":     "ReduceLROnPlateau",
                "dataset":       "ReducedMNIST",
                "train_samples": 10000,
                "test_samples":  2000
            },
            "without_attention": {
                "accuracy":        round(acc_base, 1),
                "train_time_ms":   round(tr_base,  1),
                "test_time_ms":    round(te_base,  1),
                "parameters":      params_base,
                "per_digit_acc":   per_digit_base,
                "confusion_matrix": cm_base.tolist()
            },
            "with_attention": {
                "accuracy":        round(acc_att, 1),
                "train_time_ms":   round(tr_att,  1),
                "test_time_ms":    round(te_att,  1),
                "parameters":      params_att,
                "per_digit_acc":   per_digit_att,
                "confusion_matrix": cm_att.tolist()
            }
        }
    }

    json_path = f"{RESULTS_DIR}/part_a_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved: {json_path}")

    print("\n" + "="*60)
    print("✅ Part (a) Complete!")
    print(f"  No Attention   → {acc_base:.1f}%")
    print(f"  With Attention → {acc_att:.1f}%")
    print(f"  Gain/Loss      : {acc_att - acc_base:+.1f}%")
    print("="*60)


if __name__ == "__main__":
    main()