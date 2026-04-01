
import matplotlib.pyplot as plt

import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: KV Density Stress Test (5-seed mean at kv=32)

kv_pairs = [8, 16, 32]

fixed_acc = [99.4, 99.3, 60.9]  # 5-seed means

mlp_acc   = [99.2, 99.5, 99.6]  # 5-seed means

fixed_std = [0.1,  0.2,  46.7]

mlp_std   = [0.2,  0.3,  0.1]

axes[0].errorbar(kv_pairs, fixed_acc, yerr=fixed_std, 

                fmt='b-o', label='Fixed λ', linewidth=2, markersize=8, capsize=5)

axes[0].errorbar(kv_pairs, mlp_acc, yerr=mlp_std,

                fmt='r-o', label='MLP Softplus λ', linewidth=2, markersize=8, capsize=5)

axes[0].set_xlabel('Number of KV Pairs', fontsize=12)

axes[0].set_ylabel('Val Accuracy (%)', fontsize=12)

axes[0].set_title('Complexity Stress Test\n(seq_len=128, mean ± std over 5 seeds)', fontsize=12)

axes[0].legend(fontsize=11)

axes[0].grid(True, alpha=0.3)

axes[0].set_ylim(0, 115)

axes[0].annotate('Fixed λ collapses!\n60.9% ± 46.7%', 

                xy=(32, 60.9), xytext=(22, 35),

                arrowprops=dict(arrowstyle='->', color='blue'),

                color='blue', fontsize=10)

axes[0].annotate('MLP holds!\n99.6% ± 0.1%',

                xy=(32, 99.6), xytext=(22, 82),

                arrowprops=dict(arrowstyle='->', color='red'),

                color='red', fontsize=10)

# Plot 2: Length Generalization (5-seed mean)

categories = ['Same Length\n(128→128)', 'Length Gen.\n(128→256)']

fixed_vals = [99.4, 51.0]

mlp_vals   = [99.2, 55.1]

fixed_stds = [0.1,  39.2]

mlp_stds   = [0.2,  23.8]

x = np.arange(len(categories))

width = 0.35

bars1 = axes[1].bar(x - width/2, fixed_vals, width, yerr=fixed_stds,

                    label='Fixed λ', color='blue', alpha=0.7, capsize=5)

bars2 = axes[1].bar(x + width/2, mlp_vals, width, yerr=mlp_stds,

                    label='MLP Softplus λ', color='red', alpha=0.7, capsize=5)

for bar, val in zip(bars1, fixed_vals):

    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 3,

                f'{val:.1f}%', ha='center', va='bottom', fontsize=11, color='blue')

for bar, val in zip(bars2, mlp_vals):

    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 3,

                f'{val:.1f}%', ha='center', va='bottom', fontsize=11, color='red')

axes[1].set_ylabel('Val Accuracy (%)', fontsize=12)

axes[1].set_title('Length Generalization\n(mean ± std over 5 seeds)', fontsize=12)

axes[1].set_xticks(x)

axes[1].set_xticklabels(categories, fontsize=11)

axes[1].legend(fontsize=11)

axes[1].grid(True, axis='y', alpha=0.3)

axes[1].set_ylim(0, 125)

axes[1].annotate('MLP: lower variance\n(±23.8 vs ±39.2)', 

                xy=(0.7, 55.1), xytext=(0.9, 85),

                arrowprops=dict(arrowstyle='->', color='red'),

                color='red', fontsize=10)

plt.suptitle('MLP-λ vs Fixed-λ in Log-Linear Attention\n(MQAR Benchmark, vocab=128)', 

             fontsize=13, fontweight='bold')

plt.tight_layout()

plt.savefig('mqar_results1.png', dpi=150, bbox_inches='tight')

print('Saved mqar_results.png!')

