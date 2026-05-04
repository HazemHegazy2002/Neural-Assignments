import matplotlib.pyplot as plt
import os

from step1_data import train_loader, test_loader
from step2_lenet import LeNet5
from step3_attention import LeNet5WithAttention
from step4_train import train_model

EPOCHS = 20
LR     = 0.001


def main() -> None:
    os.makedirs("./results", exist_ok=True)

    # Train LeNet-5 WITHOUT attention
    model_base = LeNet5()
    acc_base, tr_base, te_base, losses_base = train_model(
        model_base, train_loader, test_loader,
        epochs=EPOCHS, lr=LR,
        description="LeNet-5 (No Attention)"
    )

    # Train LeNet-5 WITH attention
    model_att = LeNet5WithAttention()
    acc_att, tr_att, te_att, losses_att = train_model(
        model_att, train_loader, test_loader,
        epochs=EPOCHS, lr=LR,
        description="LeNet-5 (With Spatial Attention)"
    )

    # ── COMPARISON TABLE ──
    print("\n" + "="*60)
    print("PART (a) COMPARISON TABLE")
    print("="*60)
    print(f"{'Metric':<25} {'Without Attention':>20} {'With Attention':>20}")
    print("-"*60)
    print(f"{'Accuracy (%)':<25} {acc_base:>20.1f} {acc_att:>20.1f}")
    print(f"{'Training Time (ms)':<25} {tr_base:>20.1f} {tr_att:>20.1f}")
    print(f"{'Testing Time (ms)':<25} {te_base:>20.1f} {te_att:>20.1f}")
    print("="*60)

    # ── PLOT LOSS CURVES ──
    plt.figure(figsize=(10, 5))
    plt.plot(losses_base, label="Without Attention", color="blue")
    plt.plot(losses_att,  label="With Attention",    color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Part (a) — LeNet-5 Training Loss: With vs Without Attention")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("./results/part_a_loss_curves.png")
    plt.show()
    print("✅ Step 5 Complete — Part (a) done!")


if __name__ == "__main__":
    main()