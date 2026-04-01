import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: KV Density Stress Test (seq_len=128, seed=0)
kv_pairs = [8, 16, 32]
fixed_acc = [99.5, 99.0, 60.9]
mlp_acc   = [99.1, 99.4, 99.6]

axes[0].plot(kv_pairs, fixed_acc, 'b-o', label='Fixed λ', linewidth=2, markersize=8)
axes[0].plot(kv_pairs, mlp_acc,   'r-o', label='MLP Softplus λ', linewidth=2, markersize=8)
axes[0].fill_between(kv_pairs, fixed_acc, mlp_acc, alpha=0.1, color='red')
axes[0].set_xlabel('Number of KV Pairs', fontsize=12)
axes[0].set_ylabel('Val Accuracy (%)', fontsize=12)
axes[0].set_title('Complexity Stress Test\n(seq_len=128, 5-seed mean)', fontsize=12)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(0, 105)
axes[0].annotate('Fixed λ collapses!\n60.9% ± 46.7%', 
                xy=(32, 60.9), xytext=(20, 40),
                arrowprops=dict(arrowstyle='->', color='blue'),
                color='blue', fontsize=10)
axes[0].annotate('MLP holds!\n99.6% ± 0.1%',
                xy=(32, 99.6), xytext=(20, 85),
                arrowprops=dict(arrowstyle='->', color='red'),
                color='red', fontsize=10)

# Plot 2: Length Generalization
categories = ['Same Length\n(128→128)', 'Length Generalization\n(128→256)']
fixed_vals = [99.5, 3.3]
mlp_vals   = [99.1, 33.2]
x = np.arange(len(categories))
width = 0.35

bars1 = axes[1].bar(x - width/2, fixed_vals, width, 
                     label='Fixed λ', color='blue', alpha=0.7)
bars2 = axes[1].bar(x + width/2, mlp_vals, width,
                     label='MLP Softplus λ', color='red', alpha=0.7)

# add value labels on bars
for bar in bars1:
    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=11)
for bar in bars2:
    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=11)

axes[1].set_ylabel('Val Accuracy (%)', fontsize=12)
axes[1].set_title('Length Generalization\n(trained on 128, tested on 256)', fontsize=12)
axes[1].set_xticks(x)
axes[1].set_xticklabels(categories, fontsize=10)
axes[1].legend(fontsize=11)
axes[1].grid(True, axis='y', alpha=0.3)
axes[1].set_ylim(0, 115)

plt.suptitle('MLP-λ vs Fixed-λ in Log-Linear Attention\n(MQAR Benchmark, vocab=128)', 
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('mqar_results.png', dpi=150, bbox_inches='tight')
print('Saved mqar_results.png!')
