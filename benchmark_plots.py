import json
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

# Set plot style
plt.style.use('fivethirtyeight')
sns.set_context("paper", font_scale=1.2)
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 150

# Create a directory for the plots
os.makedirs("benchmark_plots", exist_ok=True)

# Load the results
with open('benchmark_results/sequence_length_comparison.json', 'r') as f:
    results = json.load(f)

# Extract sequence lengths
seq_lengths = [int(seq_len) for seq_len in results["transformer"].keys()]
seq_lengths.sort()  # Make sure they're in ascending order

# Colors for plots
transformer_color = '#2C3E50'  # Dark blue
linformer_color = '#E74C3C'    # Red

# Plot 1: Training/validation accuracy vs. epoch for each model/sequence length
plt.figure(figsize=(16, 12))

for i, seq_len in enumerate(seq_lengths):
    plt.subplot(2, 2, i+1)
    
    # Extract epoch data
    transformer_epochs = results["transformer"][str(seq_len)]["epochs"]
    linformer_epochs = results["linformer"][str(seq_len)]["epochs"]
    
    # Extract epochs, train accuracy, and validation accuracy
    t_epochs = [ep["epoch"] for ep in transformer_epochs]
    t_train_acc = [ep["train_acc"] for ep in transformer_epochs]
    t_val_acc = [ep["val_acc"] for ep in transformer_epochs]
    
    l_epochs = [ep["epoch"] for ep in linformer_epochs]
    l_train_acc = [ep["train_acc"] for ep in linformer_epochs]
    l_val_acc = [ep["val_acc"] for ep in linformer_epochs]
    
    # Plot
    plt.plot(t_epochs, t_train_acc, '-o', color=transformer_color, label='Transformer Train')
    plt.plot(t_epochs, t_val_acc, '--o', color=transformer_color, alpha=0.7, label='Transformer Val')
    plt.plot(l_epochs, l_train_acc, '-s', color=linformer_color, label='Linformer Train')
    plt.plot(l_epochs, l_val_acc, '--s', color=linformer_color, alpha=0.7, label='Linformer Val')
    
    plt.title(f'Sequence Length = {seq_len}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0.5, 1.0)  # Adjust as needed based on your results
    plt.grid(True, alpha=0.3)
    
    if i == 0:  # Only add legend to the first subplot
        plt.legend(loc='lower right')

plt.suptitle('Training and Validation Accuracy by Epoch', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
plt.savefig('benchmark_plots/accuracy_vs_epoch.png')
plt.close()

# Plot 2: Training time vs. sequence length
plt.figure(figsize=(12, 6))

# Extract training times
transformer_times = []
linformer_times = []

for seq_len in seq_lengths:
    # Get the sum of training times across all epochs
    transformer_time = sum(ep["train_time"] for ep in results["transformer"][str(seq_len)]["epochs"])
    linformer_time = sum(ep["train_time"] for ep in results["linformer"][str(seq_len)]["epochs"])
    
    transformer_times.append(transformer_time)
    linformer_times.append(linformer_time)

# Plot
plt.plot(seq_lengths, transformer_times, '-o', linewidth=2, markersize=8, color=transformer_color, label='Transformer')
plt.plot(seq_lengths, linformer_times, '-s', linewidth=2, markersize=8, color=linformer_color, label='Linformer')

plt.title('Total Training Time vs. Sequence Length', fontsize=16)
plt.xlabel('Sequence Length')
plt.ylabel('Total Training Time (seconds)')
plt.grid(True, alpha=0.3)
plt.legend()

# Add text annotations for times
for i, seq_len in enumerate(seq_lengths):
    plt.text(seq_len, transformer_times[i] + 5, f'{transformer_times[i]:.1f}s', 
             ha='center', va='bottom', color=transformer_color)
    plt.text(seq_len, linformer_times[i] - 5, f'{linformer_times[i]:.1f}s', 
             ha='center', va='top', color=linformer_color)

plt.tight_layout()
plt.savefig('benchmark_plots/training_time_vs_seq_length.png')
plt.close()

# Plot 3: Speedup ratio vs. sequence length
plt.figure(figsize=(12, 6))

# Calculate speedup ratios (transformer time / linformer time)
speedup_ratios = [transformer_times[i] / linformer_times[i] for i in range(len(seq_lengths))]

# Create bar plot
bars = plt.bar(seq_lengths, speedup_ratios, color='#3498DB', alpha=0.8)

plt.title('Speedup Ratio (Transformer / Linformer) vs. Sequence Length', fontsize=16)
plt.xlabel('Sequence Length')
plt.ylabel('Speedup Ratio')
plt.grid(True, alpha=0.3, axis='y')

# Add text annotations for speedup ratios
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.05, f'{speedup_ratios[i]:.2f}x', 
             ha='center', va='bottom', fontweight='bold')

# Add a horizontal line at y=1 (no speedup)
plt.axhline(y=1, color='r', linestyle='--', alpha=0.7, label='No speedup')
plt.legend()

plt.tight_layout()
plt.savefig('benchmark_plots/speedup_ratio_vs_seq_length.png')
plt.close()

# Plot 4: Combined visualization - Accuracy vs Time tradeoff
plt.figure(figsize=(14, 8))

# Create scatter plot for each sequence length
for i, seq_len in enumerate(seq_lengths):
    t_summary = results["transformer"][str(seq_len)]["summary"]
    l_summary = results["linformer"][str(seq_len)]["summary"]
    
    # Plot transformer point
    plt.scatter(t_summary["total_time"], t_summary["final_val_acc"], 
                s=100 + seq_len/5, color=transformer_color, alpha=0.8, 
                marker='o', label=f'Transformer {seq_len}' if i == 0 else "")
    
    # Plot linformer point
    plt.scatter(l_summary["total_time"], l_summary["final_val_acc"], 
                s=100 + seq_len/5, color=linformer_color, alpha=0.8, 
                marker='s', label=f'Linformer {seq_len}' if i == 0 else "")
    
    # Connect the points with a line
    plt.plot([t_summary["total_time"], l_summary["total_time"]], 
             [t_summary["final_val_acc"], l_summary["final_val_acc"]], 
             'k--', alpha=0.3)
    
    # Add sequence length annotations
    plt.annotate(f'seq_len={seq_len}', 
                 xy=(t_summary["total_time"], t_summary["final_val_acc"]),
                 xytext=(10, 0), textcoords='offset points', fontsize=10)
    
    plt.annotate(f'seq_len={seq_len}', 
                 xy=(l_summary["total_time"], l_summary["final_val_acc"]),
                 xytext=(10, 0), textcoords='offset points', fontsize=10)

plt.title('Validation Accuracy vs. Total Training Time', fontsize=16)
plt.xlabel('Total Training Time (seconds)')
plt.ylabel('Final Validation Accuracy')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig('benchmark_plots/accuracy_vs_time_tradeoff.png')
plt.close()

# Plot 5: Memory usage comparison
plt.figure(figsize=(12, 6))

# Extract memory usage
transformer_memory = [results["transformer"][str(seq_len)]["summary"]["max_memory"] for seq_len in seq_lengths]
linformer_memory = [results["linformer"][str(seq_len)]["summary"]["max_memory"] for seq_len in seq_lengths]

# Set width of bars
barWidth = 0.35
r1 = np.arange(len(seq_lengths))
r2 = [x + barWidth for x in r1]

# Create grouped bars
plt.bar(r1, transformer_memory, width=barWidth, color=transformer_color, alpha=0.8, label='Transformer')
plt.bar(r2, linformer_memory, width=barWidth, color=linformer_color, alpha=0.8, label='Linformer')

# Add labels and title
plt.xlabel('Sequence Length')
plt.ylabel('Memory Usage (MB)')
plt.title('Memory Usage Comparison', fontsize=16)
plt.xticks([r + barWidth/2 for r in range(len(seq_lengths))], seq_lengths)
plt.grid(True, alpha=0.3, axis='y')
plt.legend()

# Add text annotations for memory usage
for i in range(len(seq_lengths)):
    plt.text(r1[i], transformer_memory[i] + 50, f'{transformer_memory[i]:.0f}MB', 
             ha='center', va='bottom', color=transformer_color)
    plt.text(r2[i], linformer_memory[i] + 50, f'{linformer_memory[i]:.0f}MB', 
             ha='center', va='bottom', color=linformer_color)

plt.tight_layout()
plt.savefig('benchmark_plots/memory_usage_comparison.png')
plt.close()

print("All plots have been saved to the 'benchmark_plots' directory.")