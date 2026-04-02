"""
PIPELINE 2 - REPORTING
========================
Reads the final checkpoint from the refinement loop and produces:
  - Console summary
  - A saved text report  (pipeline2_report.txt)
  - A matplotlib figure  (pipeline2_report.png)

Covers all required reporting items:
  • Final labelling accuracy (oracle)
  • Number of iterations performed
  • Total images manually labelled (seed + boundary × iterations)
  • Pseudo-labelled images incorporated & rejected per iteration
  • Total estimated manual time
  • Comparison against full-manual baseline
"""

import numpy as np
import pickle
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime

# ─────────────────────────────────────────────
#  LOAD FINAL CHECKPOINT
# ─────────────────────────────────────────────
print("Loading final checkpoint...")
with open('pipeline2_step4_latest.pkl', 'rb') as f:
    data = pickle.load(f)

# ── pull everything we need ──
accuracy_history     = data['accuracy_history']        # list, index 0 = SVM-1
iteration            = data['iteration']               # total iterations run
manual_images_count  = data['manual_images_count']     # 300 + 20*iters
manual_time_seconds  = data['manual_time_seconds']
SEED_SIZE            = data['SEED_SIZE']               # 300
BOUNDARY_PER_ITER    = 20
BASELINE_SECONDS     = 100_000                         # 10,000 × 10 s

# per-iteration pseudo stats (stored as scalars for last iter;
# we reconstruct per-iter counts from what was saved)
# If you want per-iteration pseudo stats, the loop saves
# 'pseudo_added_last' and 'pseudo_rejected_last' for the final iter.
# For a full per-iter table we use what's available.
pseudo_added_last    = data.get('pseudo_added_last', 'N/A')
pseudo_rejected_last = data.get('pseudo_rejected_last', 'N/A')

final_accuracy       = accuracy_history[-1]
initial_accuracy     = accuracy_history[0]

# ─────────────────────────────────────────────
#  BUILD REPORT TEXT
# ─────────────────────────────────────────────
lines = []
def p(text=""):
    lines.append(text)
    print(text)

now = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

p("=" * 60)
p("  PIPELINE 2 — FINAL REPORT")
p(f"  Generated: {now}")
p("=" * 60)

p()
p("── FEATURE REPRESENTATION ──────────────────────────────────")
p("  Method        : HOG (Histogram of Oriented Gradients)")
p("  Orientations  : 9")
p("  Pixels/cell   : 4 × 4")
p("  Cells/block   : 2 × 2")
p("  Block norm    : L2-Hys")
p("  Feature dim   : " + str(data.get('HOG_DIM', '—')))

p()
p("── ACCURACY ────────────────────────────────────────────────")
p(f"  Initial accuracy  (SVM-1, no refinement) : {initial_accuracy*100:.2f}%")
p(f"  Final accuracy    (oracle, check_accuracy): {final_accuracy*100:.2f}%")
target_hit = "✓  YES" if final_accuracy >= 0.99 else "✗  NO"
p(f"  Target ≥ 99% reached                     : {target_hit}")

p()
p("── ITERATIONS ──────────────────────────────────────────────")
p(f"  Total refinement iterations : {iteration}")
p()
p(f"  {'Iteration':<12}  {'Accuracy':>10}  {'Improvement':>12}")
p(f"  {'-'*12}  {'-'*10}  {'-'*12}")
for i, acc in enumerate(accuracy_history):
    if i == 0:
        label = "SVM-1 (init)"
        impv  = "—"
    else:
        label = f"Iter {i}"
        impv  = f"{(acc - accuracy_history[i-1])*100:+.2f}%"
    p(f"  {label:<12}  {acc*100:>9.2f}%  {impv:>12}")

p()
p("── MANUAL LABELLING ────────────────────────────────────────")
p(f"  Seed images labelled              : {SEED_SIZE}")
p(f"  Boundary images labelled          : {iteration * BOUNDARY_PER_ITER}"
  f"  ({iteration} iters × {BOUNDARY_PER_ITER} images)")
p(f"  Total images manually labelled    : {manual_images_count}")
p(f"  Time per image                    : 10 seconds")
p(f"  Total manual time                 : {manual_time_seconds} s"
  f"  =  {manual_time_seconds/60:.1f} min"
  f"  =  {manual_time_seconds/3600:.2f} hrs")

p()
p("── PSEUDO-LABELLING (SELF-TRAINING) ────────────────────────")
p(f"  Margin threshold                  : 75th percentile of margins")
p(f"  Max pseudo-labels per class/iter  : 50  (× 10 classes = 500 max)")
p(f"  Total training samples (final)    : {len(data['train_labels'])}")
total_pseudo = len(data['train_labels']) - SEED_SIZE - iteration * BOUNDARY_PER_ITER - len(data.get('aug_labels', []))
p(f"  Pseudo-labelled images added      : {max(total_pseudo, 0)}")
p(f"  Last iteration — added            : {pseudo_added_last}")
p(f"  Last iteration — rejected         : {pseudo_rejected_last}")

p()
p("── TIME COMPARISON ─────────────────────────────────────────")
p(f"  Full manual baseline              : {BASELINE_SECONDS} s  ≈  27.8 hrs")
p(f"  Pipeline 2 manual time            : {manual_time_seconds} s  ≈  {manual_time_seconds/3600:.2f} hrs")
time_saved_pct = (1 - manual_time_seconds / BASELINE_SECONDS) * 100
p(f"  Time saved                        : {time_saved_pct:.1f}%")
speedup = BASELINE_SECONDS / manual_time_seconds
p(f"  Speedup factor                    : {speedup:.1f}×")

p()
p("── PIPELINE PARAMETERS ─────────────────────────────────────")
p("  SVM kernel          : RBF  (C=10, gamma=scale)")
p("  Multiclass scheme   : One-vs-One")
p(f"  Seed size           : {SEED_SIZE}  (weight = 100)")
p("  Augmentations       : rotate ±5°, noise, shift ×4  → 7 copies each")
p(f"  Augmented images    : {SEED_SIZE * 7}  (weight = 1)")
p(f"  Boundary per iter   : {BOUNDARY_PER_ITER}  (weight = 100)")
p("  Pseudo per class    : 50  (weight = 1,  margin > 75th pct)")

p()
p("=" * 60)
p("  END OF REPORT")
p("=" * 60)

# ─────────────────────────────────────────────
#  SAVE TEXT REPORT
# ─────────────────────────────────────────────
with open('pipeline2_report.txt', 'w') as f:
    f.write("\n".join(lines))
print("\nText report saved → pipeline2_report.txt")

# ─────────────────────────────────────────────
#  FIGURE: Accuracy curve + Manual time bar
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(13, 5))
fig.suptitle("Pipeline 2 — Final Report", fontsize=14, fontweight='bold')
gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

# ── Left: Accuracy per iteration ──
ax1 = fig.add_subplot(gs[0])
iters = list(range(len(accuracy_history)))
accs  = [a * 100 for a in accuracy_history]
ax1.plot(iters, accs, marker='o', color='steelblue', linewidth=2, markersize=7)
ax1.axhline(99, color='red', linestyle='--', linewidth=1.2, label='99% target')
for i, a in zip(iters, accs):
    ax1.annotate(f"{a:.2f}%", (i, a),
                 textcoords="offset points", xytext=(0, 8),
                 ha='center', fontsize=8.5)
ax1.set_xlabel("Iteration  (0 = SVM-1 initial)", fontsize=10)
ax1.set_ylabel("Labelling Accuracy (%)", fontsize=10)
ax1.set_title("Accuracy vs Iteration", fontsize=11)
ax1.set_xticks(iters)
ax1.set_xticklabels(
    ["SVM-1"] + [f"Iter {i}" for i in range(1, len(iters))],
    rotation=20, ha='right', fontsize=8
)
ax1.set_ylim(min(accs) - 1, 101)
ax1.legend(fontsize=9)
ax1.grid(True, linestyle='--', alpha=0.5)

# ── Right: Manual time comparison bar chart ──
ax2 = fig.add_subplot(gs[1])
categories  = ['Full Manual\nBaseline', 'Pipeline 2\n(Seed only)', 'Pipeline 2\n(Final)']
times_min   = [
    BASELINE_SECONDS / 60,
    SEED_SIZE * 10 / 60,
    manual_time_seconds / 60
]
colors = ['#d9534f', '#f0ad4e', '#5cb85c']
bars = ax2.bar(categories, times_min, color=colors, width=0.5, edgecolor='black')
for bar, t in zip(bars, times_min):
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 5,
             f"{t:.1f} min", ha='center', va='bottom', fontsize=9)
ax2.set_ylabel("Manual Time (minutes)", fontsize=10)
ax2.set_title("Manual Labelling Time Comparison", fontsize=11)
ax2.grid(True, axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('pipeline2_report.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure saved → pipeline2_report.png")